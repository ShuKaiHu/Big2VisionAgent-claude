from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page

from big2_vision_agent.browser.selectors import ACTION_HINT_TEXTS, LOGIN_HINT_TEXTS
from big2_vision_agent.config import Settings


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class StorageSnapshot:
    cookies: list[dict]
    local_storage: dict[str, str]
    session_storage: dict[str, str]


@dataclass(slots=True)
class NetworkEntry:
    method: str
    resource_type: str
    url: str


@dataclass(slots=True)
class RuntimeInfo:
    has_cc: bool
    current_scene: str | None
    frame_size: dict[str, float] | None
    visible_size: dict[str, float] | None
    design_resolution: dict[str, float] | None
    pskey: str | None


@dataclass(slots=True)
class SceneNode:
    name: str
    active: bool
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    text: str | None
    children: list["SceneNode"]


@dataclass(slots=True)
class PageSummary:
    url: str
    title: str
    frame_urls: list[str]
    canvas_count: int
    canvas_box: BoundingBox | None
    button_like_count: int
    input_count: int
    login_hints: list[str]
    action_hints: list[str]
    body_text_sample: str
    global_keys: list[str]
    runtime: RuntimeInfo
    storage: StorageSnapshot
    recent_requests: list[NetworkEntry]


async def inspect_page(page: Page, settings: Settings) -> Path:
    output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = await _collect_summary(page)
    await page.screenshot(path=str(output_dir / "page.png"), full_page=True)
    html = await page.content()
    (output_dir / "page.html").write_text(html, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await save_network_log(page, output_dir)
    return output_dir


async def read_network_log(page: Page) -> list[dict]:
    return await page.evaluate(
        """
        () => Array.isArray(window.__big2NetworkLog)
          ? window.__big2NetworkLog.slice()
          : []
        """
    )


async def save_network_log(page: Page, output_dir: Path, filename: str = "network_log.json") -> Path:
    logs = await read_network_log(page)
    path = output_dir / filename
    path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _collect_summary(page: Page) -> PageSummary:
    recent_requests: list[NetworkEntry] = []

    def handle_request(request) -> None:
        if len(recent_requests) >= 40:
            return
        recent_requests.append(
            NetworkEntry(
                method=request.method,
                resource_type=request.resource_type,
                url=request.url,
            )
        )

    page.on("request", handle_request)
    await page.wait_for_timeout(3000)

    title = await page.title()
    frame_urls = [frame.url for frame in page.frames]
    button_like_count = await page.locator("button, [role='button'], a").count()
    canvas_count = await page.locator("canvas").count()
    input_count = await page.locator("input, textarea, select").count()
    body_text = await page.locator("body").inner_text()
    canvas_box = await _read_canvas_box(page)
    global_keys = await _read_global_keys(page)
    runtime = await _read_runtime(page)
    storage = await _read_storage(page)

    return PageSummary(
        url=page.url,
        title=title,
        frame_urls=frame_urls,
        canvas_count=canvas_count,
        canvas_box=canvas_box,
        button_like_count=button_like_count,
        input_count=input_count,
        login_hints=_find_hints(body_text, LOGIN_HINT_TEXTS),
        action_hints=_find_hints(body_text, ACTION_HINT_TEXTS),
        body_text_sample=body_text[:1000],
        global_keys=global_keys,
        runtime=runtime,
        storage=storage,
        recent_requests=recent_requests,
    )


def _find_hints(body_text: str, hints: tuple[str, ...]) -> list[str]:
    lower_text = body_text.lower()
    return [hint for hint in hints if hint.lower() in lower_text]


async def _read_canvas_box(page: Page) -> BoundingBox | None:
    canvas = page.locator("canvas").first
    if await canvas.count() == 0:
        return None
    box = await canvas.bounding_box()
    if box is None:
        return None
    return BoundingBox(
        x=box["x"],
        y=box["y"],
        width=box["width"],
        height=box["height"],
    )


async def _read_global_keys(page: Page) -> list[str]:
    keys = await page.evaluate(
        """
        () => Object.keys(window)
          .filter((key) => /^(cc|Cocos|game|Game|socket|Socket|PSKEY|io)$/i.test(key))
          .sort()
        """
    )
    return keys


async def _read_storage(page: Page) -> StorageSnapshot:
    local_storage = await page.evaluate(
        "() => Object.fromEntries(Object.entries(window.localStorage))"
    )
    session_storage = await page.evaluate(
        "() => Object.fromEntries(Object.entries(window.sessionStorage))"
    )
    cookies = await page.context.cookies()
    return StorageSnapshot(
        cookies=cookies,
        local_storage=local_storage,
        session_storage=session_storage,
    )


async def _read_runtime(page: Page) -> RuntimeInfo:
    runtime = await page.evaluate(
        """
        () => {
          const w = window;
          const result = {
            has_cc: Boolean(w.cc),
            current_scene: null,
            frame_size: null,
            visible_size: null,
            design_resolution: null,
            pskey: typeof w.PSKEY === 'string' ? w.PSKEY : null,
          };

          if (!w.cc) {
            return result;
          }

          try {
            const scene = w.cc.director && w.cc.director.getScene ? w.cc.director.getScene() : null;
            result.current_scene = scene && scene.name ? scene.name : null;
          } catch (error) {
            result.current_scene = null;
          }

          try {
            const frameSize = w.cc.view && w.cc.view.getFrameSize ? w.cc.view.getFrameSize() : null;
            if (frameSize) {
              result.frame_size = { width: frameSize.width, height: frameSize.height };
            }
          } catch (error) {}

          try {
            const visibleSize = w.cc.view && w.cc.view.getVisibleSize ? w.cc.view.getVisibleSize() : null;
            if (visibleSize) {
              result.visible_size = { width: visibleSize.width, height: visibleSize.height };
            }
          } catch (error) {}

          try {
            const resolution = w.cc.view && w.cc.view.getDesignResolutionSize ? w.cc.view.getDesignResolutionSize() : null;
            if (resolution) {
              result.design_resolution = { width: resolution.width, height: resolution.height };
            }
          } catch (error) {}

          return result;
        }
        """
    )
    return RuntimeInfo(**runtime)


async def dump_scene_tree(page: Page) -> dict:
    return await page.evaluate(
        """
        () => {
          function readNode(node, depth, maxDepth) {
            if (!node || depth > maxDepth) {
              return null;
            }

            let text = null;
            try {
              const label = node.getComponent && (node.getComponent('cc.Label') || node.getComponent(cc.Label));
              if (label && typeof label.string === 'string') {
                text = label.string;
              }
            } catch (error) {}

            const result = {
              name: node.name || '',
              active: Boolean(node.active),
              x: typeof node.x === 'number' ? node.x : null,
              y: typeof node.y === 'number' ? node.y : null,
              width: typeof node.width === 'number' ? node.width : null,
              height: typeof node.height === 'number' ? node.height : null,
              text,
              children: [],
            };

            const children = Array.isArray(node.children) ? node.children : [];
            for (const child of children) {
              const childResult = readNode(child, depth + 1, maxDepth);
              if (childResult) {
                result.children.push(childResult);
              }
            }
            return result;
          }

          const scene = window.cc && window.cc.director && window.cc.director.getScene
            ? window.cc.director.getScene()
            : null;
          if (!scene) {
            return {};
          }

          return readNode(scene, 0, 8);
        }
        """
    )
