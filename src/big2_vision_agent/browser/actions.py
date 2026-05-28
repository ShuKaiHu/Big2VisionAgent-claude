from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Page

DEFAULT_DESIGN_WIDTH = 1728.0
DEFAULT_DESIGN_HEIGHT = 1152.0


@dataclass(slots=True)
class CanvasGeometry:
    x: float
    y: float
    width: float
    height: float
    design_width: float
    design_height: float


def resolve_live_page(page: Page) -> Page:
    try:
        if not page.is_closed():
            return page
    except Exception:
        pass
    try:
        live_pages = [candidate for candidate in page.context.pages if not candidate.is_closed()]
        if live_pages:
            return live_pages[-1]
    except Exception:
        return page
    return page


async def read_canvas_geometry(page: Page) -> CanvasGeometry:
    box = None
    for _ in range(30):
        page = resolve_live_page(page)
        await page.wait_for_timeout(500)
        try:
            box = await page.evaluate(
                """
                () => {
                  const canvas = document.querySelector('canvas');
                  if (!canvas) {
                    return null;
                  }
                  const rect = canvas.getBoundingClientRect();
                  return {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                  };
                }
                """
            )
        except Exception:
            box = None
        if box is not None:
            break
    if box is None:
        raise RuntimeError("Canvas bounding box is not available.")

    runtime = await _read_design_resolution(page)
    if runtime is None:
        runtime = {"width": DEFAULT_DESIGN_WIDTH, "height": DEFAULT_DESIGN_HEIGHT}

    return CanvasGeometry(
        x=box["x"],
        y=box["y"],
        width=box["width"],
        height=box["height"],
        design_width=runtime["width"],
        design_height=runtime["height"],
    )


async def _read_design_resolution(page: Page) -> dict[str, float] | None:
    for _ in range(10):
        page = resolve_live_page(page)
        runtime = await page.evaluate(
        """
        () => {
          const ccView = window.cc && window.cc.view;
          const resolution = ccView && ccView.getDesignResolutionSize
            ? ccView.getDesignResolutionSize()
            : null;
          return resolution
            ? { width: resolution.width, height: resolution.height }
            : null;
        }
        """
        )
        if runtime is not None:
            return runtime
        await page.wait_for_timeout(300)
    return None


async def click_canvas_design_position(page: Page, design_x: float, design_y: float) -> dict[str, float]:
    page = resolve_live_page(page)
    geometry = await read_canvas_geometry(page)
    screen_x = geometry.x + (design_x / geometry.design_width) * geometry.width
    screen_y = geometry.y + (design_y / geometry.design_height) * geometry.height
    await page.mouse.click(screen_x, screen_y)
    return {
        "design_x": design_x,
        "design_y": design_y,
        "screen_x": screen_x,
        "screen_y": screen_y,
    }


async def read_current_scene(page: Page) -> str | None:
    return await page.evaluate(
        """
        () => {
          try {
            const scene = window.cc && window.cc.director && window.cc.director.getScene
              ? window.cc.director.getScene()
              : null;
            return scene && scene.name ? scene.name : null;
          } catch (error) {
            return null;
          }
        }
        """
    )


async def probe_nodes_by_name(page: Page, name: str) -> list[dict]:
    return await probe_nodes(page, name, exact=True)


async def probe_nodes(page: Page, name: str, exact: bool = False) -> list[dict]:
    return await page.evaluate(
        """
        ({ targetName, exact }) => {
          const result = [];
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          const view = ccGlobal && ccGlobal.view && ccGlobal.view.getDesignResolutionSize
            ? ccGlobal.view.getDesignResolutionSize()
            : null;
          if (!scene || !view) {
            return result;
          }

          function componentNames(node) {
            try {
              return (node._components || [])
                .map((component) => {
                  if (component && component.__classname__) {
                    return component.__classname__;
                  }
                  const ctor = component && component.constructor;
                  return ctor && ctor.name ? ctor.name : 'UnknownComponent';
                });
            } catch (error) {
              return [];
            }
          }

          function readLabel(node) {
            try {
              const label = node.getComponent && (node.getComponent('cc.Label') || node.getComponent(ccGlobal.Label));
              return label && typeof label.string === 'string' ? label.string : null;
            } catch (error) {
              return null;
            }
          }

          function visit(node, path) {
            const nextPath = [...path, node.name || ''];
            const matched = exact
              ? node.name === targetName
              : Boolean(node.name && node.name.includes(targetName));
            if (matched) {
              let rect = null;
              try {
                const box = node.getBoundingBoxToWorld();
                rect = {
                  x: box.x,
                  y: box.y,
                  width: box.width,
                  height: box.height,
                  center_x: box.x + box.width / 2,
                  center_y_bottom_left: box.y + box.height / 2,
                  center_y_top_left: view.height - (box.y + box.height / 2),
                };
              } catch (error) {}

              result.push({
                path: nextPath.join(' / '),
                active: Boolean(node.active),
                x: typeof node.x === 'number' ? node.x : null,
                y: typeof node.y === 'number' ? node.y : null,
                width: typeof node.width === 'number' ? node.width : null,
                height: typeof node.height === 'number' ? node.height : null,
                text: readLabel(node),
                components: componentNames(node),
                rect,
              });
            }

            const children = Array.isArray(node.children) ? node.children : [];
            for (const child of children) {
              visit(child, nextPath);
            }
          }

          visit(scene, []);
          return result;
        }
        """,
        {"targetName": name, "exact": exact},
    )


async def click_named_node(page: Page, name: str, occurrence: int = 0) -> dict[str, float | str | int | None]:
    matches = await probe_nodes_by_name(page, name)
    active_matches = [match for match in matches if match.get("active") and match.get("rect")]
    if occurrence >= len(active_matches):
        raise RuntimeError(
            f"Node '{name}' occurrence {occurrence} is unavailable. Active matches: {len(active_matches)}."
        )

    match = active_matches[occurrence]
    rect = match["rect"]
    center_x = rect["center_x"]
    center_y_top_left = rect["center_y_top_left"]
    click_result = await click_canvas_design_position(page, center_x, center_y_top_left)
    click_result["path"] = match["path"]
    click_result["occurrence"] = occurrence
    return click_result


async def invoke_named_node(page: Page, name: str, occurrence: int = 0) -> dict:
    return await invoke_node(page, name, occurrence=occurrence, exact=True)


async def invoke_node(page: Page, name: str, occurrence: int = 0, exact: bool = False) -> dict:
    return await page.evaluate(
        """
        ({ targetName, occurrence, exact }) => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          if (!scene) {
            throw new Error('Scene is unavailable.');
          }

          const matches = [];
          function visit(node, path) {
            const nextPath = [...path, node.name || ''];
            const matched = exact
              ? node.name === targetName
              : Boolean(node.name && node.name.includes(targetName));
            if (matched && node.active) {
                matches.push({ node, path: nextPath.join(' / ') });
            }
            const children = Array.isArray(node.children) ? node.children : [];
            for (const child of children) {
              visit(child, nextPath);
            }
          }
          visit(scene, []);

          if (occurrence >= matches.length) {
            throw new Error(`Node '${targetName}' occurrence ${occurrence} not found.`);
          }

          const match = matches[occurrence];
          const node = match.node;
          const button = node.getComponent && (node.getComponent('cc.Button') || node.getComponent(ccGlobal.Button));

          let invoked = false;
          let mode = 'none';

          if (button && Array.isArray(button.clickEvents) && button.clickEvents.length > 0) {
            for (const eventHandler of button.clickEvents) {
              ccGlobal.Component && ccGlobal.Component.EventHandler.emitEvents([eventHandler], { type: 'click' });
            }
            invoked = true;
            mode = 'clickEvents';
          }

          if (!invoked) {
            try {
              node.emit('click');
              node.emit('touchend');
              invoked = true;
              mode = 'emit';
            } catch (error) {}
          }

          return {
            path: match.path,
            invoked,
            mode,
            has_button: Boolean(button),
            component_names: (node._components || []).map((component) => component && (component.__classname__ || (component.constructor && component.constructor.name) || 'UnknownComponent')),
          };
        }
        """,
        {"targetName": name, "occurrence": occurrence, "exact": exact},
    )


async def toggle_my_card_by_sprite(page: Page, sprite_frame: str) -> dict:
    return await page.evaluate(
        """
        ({ targetSprite }) => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          if (!scene) {
            throw new Error('Scene is unavailable.');
          }

          function byName(node, name) {
            return (node && Array.isArray(node.children) ? node.children : []).find((child) => child.name === name) || null;
          }

          function pathNode(path) {
            let node = scene;
            for (const name of path) {
              node = byName(node, name);
              if (!node) {
                return null;
              }
            }
            return node;
          }

          function spriteFrameName(node) {
            if (!node) return null;
            try {
              const sprite = node.getComponent && (node.getComponent('cc.Sprite') || node.getComponent(ccGlobal.Sprite));
              const frame = sprite && sprite.spriteFrame ? sprite.spriteFrame : null;
              if (!frame) return null;
              return frame.name || frame._name || null;
            } catch (error) {
              return null;
            }
          }

          const wall = pathNode(['GameScene', 'GameLayer', 'CardLayer', 'MyCardWallLayer']);
          if (!wall) {
            throw new Error('MyCardWallLayer is unavailable.');
          }

          const cards = [];
          for (const child of wall.children || []) {
            if (child.name !== 'Card' || !child.activeInHierarchy) continue;
            const foreSprite = byName(child, 'ForeSprite');
            const selectFrame = byName(child, 'SelectFrame');
            cards.push({
              node: child,
              sprite_frame: spriteFrameName(foreSprite),
              selected: Boolean(selectFrame && selectFrame.activeInHierarchy),
              component_names: (child._components || []).map((component) => component && (component.__classname__ || (component.constructor && component.constructor.name) || 'UnknownComponent')),
            });
          }

          const match = cards.find((card) => card.sprite_frame === targetSprite);
          if (!match) {
            return {
              invoked: false,
              reason: 'card_not_found',
              target_sprite: targetSprite,
              cards: cards.map((card) => ({
                sprite_frame: card.sprite_frame,
                selected: card.selected,
                component_names: card.component_names,
              })),
            };
          }

          const node = match.node;
          const cardComponent = node.getComponent && node.getComponent('Card');
          const button = node.getComponent && (node.getComponent('cc.Button') || node.getComponent(ccGlobal.Button));
          const eventTypes = [];
          const nodeEventType = ccGlobal.Node && ccGlobal.Node.EventType ? ccGlobal.Node.EventType : null;
          const methodCalls = [];

          function isSelected() {
            const frame = byName(node, 'SelectFrame');
            return Boolean(frame && frame.activeInHierarchy);
          }

          function invokeButtonClickEvents(buttonComponent) {
            const invoked = [];
            try {
              const events = Array.isArray(buttonComponent && buttonComponent.clickEvents)
                ? buttonComponent.clickEvents
                : [];
              for (const clickEvent of events) {
                try {
                  if (!clickEvent) continue;
                  const target = clickEvent.target || null;
                  const componentName = clickEvent.component || null;
                  const handler = clickEvent.handler || null;
                  if (!target || !componentName || !handler) continue;
                  const component = target.getComponent && target.getComponent(componentName);
                  if (!component || typeof component[handler] !== 'function') continue;
                  component[handler](node);
                  invoked.push(`${componentName}.${handler}`);
                } catch (error) {}
              }
            } catch (error) {}
            return invoked;
          }

          if (cardComponent && typeof cardComponent.setSelect === 'function') {
            try {
              cardComponent.setSelect(true);
              methodCalls.push('Card.setSelect(true)');
            } catch (error) {}
          }
          if (cardComponent && typeof cardComponent.setLight === 'function') {
            try {
              cardComponent.setLight(true);
              methodCalls.push('Card.setLight(true)');
            } catch (error) {}
          }
          if (!isSelected() && cardComponent && typeof cardComponent.setDisabled === 'function') {
            try {
              cardComponent.setDisabled(false);
              methodCalls.push('Card.setDisabled(false)');
            } catch (error) {}
          }
          if (!isSelected() && button) {
            const clickInvoked = invokeButtonClickEvents(button);
            methodCalls.push(...clickInvoked.map((name) => `button:${name}`));
          }

          const selectedAfterMethod = isSelected();

          if (selectedAfterMethod) {
            return {
              invoked: true,
              emitted: [],
              target_sprite: targetSprite,
              component_names: match.component_names,
              selected_before: match.selected,
              method_calls: methodCalls,
              mode: 'component_method',
            };
          }

          if (nodeEventType) {
            eventTypes.push(nodeEventType.TOUCH_START, nodeEventType.TOUCH_END);
          }
          eventTypes.push('touchstart', 'touchend', 'click');

          let emitted = [];
          for (const eventType of eventTypes) {
            if (!eventType) continue;
            try {
              node.emit(eventType, {
                type: eventType,
                target: node,
                currentTarget: node,
              });
              emitted.push(eventType);
            } catch (error) {}
          }

          return {
            invoked: emitted.length > 0,
            emitted,
            target_sprite: targetSprite,
            component_names: match.component_names,
            selected_before: match.selected,
            method_calls: methodCalls,
            mode: emitted.length > 0 ? 'emit' : 'none',
          };
        }
        """,
        {"targetSprite": sprite_frame},
    )


async def deselect_all_selected_cards(page) -> dict:
    """Call setSelect(false) on every currently-selected card via Cocos API.

    This is the reliable path for clearing stale selections — unlike pixel clicks
    it is not confused by overlapping cards, and unlike toggle_my_card_by_sprite
    (which always calls setSelect(true)) it actually deselects.
    """
    return await page.evaluate(
        """
        () => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          if (!scene) return { ok: false, reason: 'no_scene', deselected: [] };

          function byName(node, name) {
            return (node && Array.isArray(node.children) ? node.children : [])
              .find((child) => child.name === name) || null;
          }

          function pathNode(path) {
            let node = scene;
            for (const name of path) {
              node = byName(node, name);
              if (!node) return null;
            }
            return node;
          }

          function isSelected(cardNode) {
            const frame = byName(cardNode, 'SelectFrame');
            return Boolean(frame && frame.activeInHierarchy);
          }

          const wall = pathNode(['GameScene', 'GameLayer', 'CardLayer', 'MyCardWallLayer']);
          if (!wall) return { ok: false, reason: 'no_wall', deselected: [] };

          const deselected = [];
          for (const child of wall.children || []) {
            if (child.name !== 'Card' || !child.activeInHierarchy) continue;
            if (!isSelected(child)) continue;
            const cardComponent = child.getComponent && child.getComponent('Card');
            if (cardComponent && typeof cardComponent.setSelect === 'function') {
              try {
                cardComponent.setSelect(false);
                deselected.push(child.name);
              } catch (e) {}
            }
          }
          return { ok: true, deselected };
        }
        """
    )


async def inspect_my_card_by_sprite(page: Page, sprite_frame: str) -> dict:
    return await page.evaluate(
        """
        ({ targetSprite }) => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          if (!scene) {
            throw new Error('Scene is unavailable.');
          }

          function byName(node, name) {
            return (node && Array.isArray(node.children) ? node.children : []).find((child) => child.name === name) || null;
          }

          function pathNode(path) {
            let node = scene;
            for (const name of path) {
              node = byName(node, name);
              if (!node) {
                return null;
              }
            }
            return node;
          }

          function spriteFrameName(node) {
            if (!node) return null;
            try {
              const sprite = node.getComponent && (node.getComponent('cc.Sprite') || node.getComponent(ccGlobal.Sprite));
              const frame = sprite && sprite.spriteFrame ? sprite.spriteFrame : null;
              if (!frame) return null;
              return frame.name || frame._name || null;
            } catch (error) {
              return null;
            }
          }

          function listMethods(component) {
            const names = new Set();
            let proto = component ? Object.getPrototypeOf(component) : null;
            let depth = 0;
            while (proto && proto !== Object.prototype && depth < 4) {
              for (const name of Object.getOwnPropertyNames(proto)) {
                if (name === 'constructor' || name.startsWith('_')) continue;
                const value = component[name];
                if (typeof value === 'function') {
                  names.add(name);
                }
              }
              proto = Object.getPrototypeOf(proto);
              depth += 1;
            }
            return [...names].sort();
          }

          const wall = pathNode(['GameScene', 'GameLayer', 'CardLayer', 'MyCardWallLayer']);
          if (!wall) {
            throw new Error('MyCardWallLayer is unavailable.');
          }

          for (const child of wall.children || []) {
            if (child.name !== 'Card' || !child.activeInHierarchy) continue;
            const foreSprite = byName(child, 'ForeSprite');
            const target = spriteFrameName(foreSprite);
            if (target !== targetSprite) continue;
            const components = (child._components || []).filter(Boolean).map((component) => ({
              name: component.__classname__ || (component.constructor && component.constructor.name) || 'UnknownComponent',
              methods: listMethods(component),
              own_keys: Object.keys(component).sort(),
              function_arity: Object.fromEntries(
                listMethods(component).map((method) => {
                  try {
                    const value = component[method];
                    return [method, typeof value === 'function' ? value.length : null];
                  } catch (error) {
                    return [method, null];
                  }
                })
              ),
            }));
            const button = child.getComponent && (child.getComponent('cc.Button') || child.getComponent(ccGlobal.Button));
            return {
              found: true,
              sprite_frame: target,
              node_name: child.name,
              selected: Boolean(byName(child, 'SelectFrame') && byName(child, 'SelectFrame').activeInHierarchy),
              node_keys: Object.keys(child).sort(),
              button_click_events: Array.isArray(button && button.clickEvents)
                ? button.clickEvents.map((clickEvent) => ({
                    component: clickEvent && clickEvent.component ? clickEvent.component : null,
                    handler: clickEvent && clickEvent.handler ? clickEvent.handler : null,
                    target_name: clickEvent && clickEvent.target ? clickEvent.target.name || null : null,
                  }))
                : [],
              components,
            };
          }

          return {
            found: false,
            sprite_frame: targetSprite,
          };
        }
        """,
        {"targetSprite": sprite_frame},
    )


async def read_big2_game_state(page: Page) -> dict:
    return await page.evaluate(
        """
        () => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          const view = ccGlobal && ccGlobal.view && ccGlobal.view.getDesignResolutionSize
            ? ccGlobal.view.getDesignResolutionSize()
            : null;
          if (!scene || !view) {
            return { scene: null };
          }

          function byName(node, name) {
            return (node && Array.isArray(node.children) ? node.children : []).find((child) => child.name === name) || null;
          }

          function pathNode(path) {
            let node = scene;
            for (const name of path) {
              node = byName(node, name);
              if (!node) {
                return null;
              }
            }
            return node;
          }

          function worldCenter(node) {
            if (!node || !node.getBoundingBoxToWorld) {
              return null;
            }
            const box = node.getBoundingBoxToWorld();
            return {
              x: box.x + box.width / 2,
              y: view.height - (box.y + box.height / 2),
            };
          }

          function worldBox(node) {
            if (!node || !node.getBoundingBoxToWorld) {
              return null;
            }
            const box = node.getBoundingBoxToWorld();
            return {
              left: box.x,
              right: box.x + box.width,
              top: view.height - (box.y + box.height),
              bottom: view.height - box.y,
              width: box.width,
              height: box.height,
            };
          }

          function labelText(node) {
            if (!node) return null;
            try {
              const label = node.getComponent && (node.getComponent('cc.Label') || node.getComponent(ccGlobal.Label));
              return label && typeof label.string === 'string' ? label.string : null;
            } catch (error) {
              return null;
            }
          }

          function spriteFrameName(node) {
            if (!node) return null;
            try {
              const sprite = node.getComponent && (node.getComponent('cc.Sprite') || node.getComponent(ccGlobal.Sprite));
              const frame = sprite && sprite.spriteFrame ? sprite.spriteFrame : null;
              if (!frame) return null;
              return frame.name || frame._name || null;
            } catch (error) {
              return null;
            }
          }

          const gameLayer = pathNode(['GameScene', 'GameLayer']);
          if (!gameLayer) {
            return { scene: scene.name || null };
          }

          const cardLayer = byName(gameLayer, 'CardLayer');
          const myCardWallLayer = pathNode(['GameScene', 'GameLayer', 'CardLayer', 'MyCardWallLayer']);
          const myProfileLayer = byName(gameLayer, 'MyProfileLayer');
          const actionLayer = byName(gameLayer, 'ActionLayer');
          const changeCardLayer = byName(gameLayer, 'ChangeCardLayer');
          const enemyProfileLayer = byName(gameLayer, 'EnemyProfileLayer');
          const hintArrowLayer = byName(gameLayer, 'HintArrowLayer');
          const showCardLayer = byName(gameLayer, 'ShowCardLayer');

          const myCards = [];
          for (const child of (myCardWallLayer && myCardWallLayer.children) || []) {
            if (child.name !== 'Card' || !child.activeInHierarchy) continue;
            const disableMask = byName(child, 'DisableMask');
            const selectFrame = byName(child, 'SelectFrame');
            const foreSprite = byName(child, 'ForeSprite');
            myCards.push({
              center: worldCenter(child),
              box: worldBox(child),
              disabled: Boolean(disableMask && disableMask.activeInHierarchy),
              selected: Boolean(selectFrame && selectFrame.activeInHierarchy),
              face_up: Boolean(foreSprite && foreSprite.activeInHierarchy),
              sprite_frame: spriteFrameName(foreSprite),
            });
          }

          const enemyProfiles = [];
          for (const child of (enemyProfileLayer && enemyProfileLayer.children) || []) {
            if (child.name !== 'EnemyProfile' || !child.activeInHierarchy) continue;
            const remain = byName(child, 'RemainCardCount');
            const remainInner = remain && byName(remain, 'RemainCardCount');
            const backgroundLight = byName(child, 'BackgroundLight');
            enemyProfiles.push({
              center: worldCenter(child),
              remain_text: labelText(remainInner),
              highlighted: Boolean(backgroundLight && backgroundLight.activeInHierarchy),
            });
          }

          function actionButton(name) {
            const layout = actionLayer && byName(actionLayer, 'ActionButtonLayout');
            const node = layout && byName(layout, name);
            let interactable = null;
            try {
              const button = node && node.getComponent && (node.getComponent('cc.Button') || node.getComponent(ccGlobal.Button));
              interactable = button ? Boolean(button.interactable) : null;
            } catch (error) {}
            return {
              active: Boolean(node && node.activeInHierarchy),
              center: worldCenter(node),
              interactable,
            };
          }

          function readButtonState(node) {
            if (!node) {
              return {
                active: false,
                interactable: null,
                center: null,
                opacity: null,
                color: null,
                scale_x: null,
                scale_y: null,
                text: null,
              };
            }

            let interactable = null;
            try {
              const button = node.getComponent && (node.getComponent('cc.Button') || node.getComponent(ccGlobal.Button));
              interactable = button ? Boolean(button.interactable) : null;
            } catch (error) {}

            return {
              active: Boolean(node.activeInHierarchy),
              interactable,
              center: worldCenter(node),
              opacity: typeof node.opacity === 'number' ? node.opacity : null,
              color: node.color ? { r: node.color.r, g: node.color.g, b: node.color.b } : null,
              scale_x: typeof node.scaleX === 'number' ? node.scaleX : null,
              scale_y: typeof node.scaleY === 'number' ? node.scaleY : null,
              text: labelText(byName(node, 'Label')) || labelText(node),
            };
          }

          function readCardNode(node) {
            if (!node || !node.activeInHierarchy) {
              return null;
            }
            const foreSprite = byName(node, 'ForeSprite') || node;
            return {
              name: node.name,
              center: worldCenter(node),
              width: typeof node.width === 'number' ? node.width : null,
              height: typeof node.height === 'number' ? node.height : null,
              sprite_frame: spriteFrameName(foreSprite),
            };
          }

          function readShowCardSet(cardSet) {
            if (!cardSet || !cardSet.activeInHierarchy) {
              return null;
            }

            const layout = byName(cardSet, 'CardLayout_LayoutOff');
            const cards = [];
            for (const child of (layout && layout.children) || []) {
              if (!child.name || !child.name.startsWith('Card')) continue;
              const card = readCardNode(child);
              if (card) {
                cards.push(card);
              }
            }

            const cardTypeSprite = byName(cardSet, 'CardType_Sprite');
            const typeDecoration = byName(cardSet, 'TypeDecoration');

            return {
              center: worldCenter(cardSet),
              card_count: cards.length,
              cards,
              card_type_sprite_active: Boolean(cardTypeSprite && cardTypeSprite.activeInHierarchy),
              card_type_sprite_frame: spriteFrameName(cardTypeSprite),
              decoration_active: Boolean(typeDecoration && typeDecoration.activeInHierarchy),
            };
          }

          function collectVisibleShowCardSets(root) {
            const sets = [];
            function visit(node) {
              if (!node || !node.activeInHierarchy) {
                return;
              }
              if (node.name === 'CardSet') {
                const cardSet = readShowCardSet(node);
                if (cardSet && cardSet.card_count > 0) {
                  sets.push(cardSet);
                }
              }
              const children = Array.isArray(node.children) ? node.children : [];
              for (const child of children) {
                visit(child);
              }
            }
            visit(root);
            return sets;
          }

          const myGameClock = myProfileLayer && byName(myProfileLayer, 'GameClock');
          const enemyHourglass = enemyProfileLayer && byName(enemyProfileLayer, 'HourglassNode');
          const hintArrow = hintArrowLayer && byName(hintArrowLayer, 'Arrow');
          const changeConfirm = changeCardLayer && byName(changeCardLayer, 'ConfirmBtn');
          const changeClock = changeCardLayer && byName(changeCardLayer, 'GameClock');
          const cardTypeButtonLayout = actionLayer && byName(actionLayer, 'CardTypeButtonLayout');
          const gameInfo = pathNode(['GameScene', 'GameLayer', 'MyProfileSetting', 'GameInfo']);
          const gameType = gameInfo && byName(gameInfo, 'GameType');
          const round = gameInfo && byName(gameInfo, 'Round');
          const systemMsgLayer = byName(gameLayer, 'SystemMsgLayer');

          const cardTypeButtons = {};
          const cardTypeMap = {
            single: 'CardTypeButton-Single',
            pair: 'CardTypeButton-Pair',
            straight: 'CardTypeButton-Straight',
            full_house: 'CardTypeButton-FullHouse',
            four_kind: 'CardTypeButton-FourKind',
            straight_flush: 'CardTypeButton-StraightFlush',
          };
          for (const [key, nodeName] of Object.entries(cardTypeMap)) {
            cardTypeButtons[key] = readButtonState(cardTypeButtonLayout && byName(cardTypeButtonLayout, nodeName));
          }

          const visibleCardTypeButtons = Object.entries(cardTypeButtons)
            .filter(([, value]) => value.active)
            .map(([key, value]) => ({ key, ...value }));

          const activeTypeCandidates = visibleCardTypeButtons.filter((value) => value.interactable !== false);

          const systemMessages = {
            card_type_error: Boolean(systemMsgLayer && byName(systemMsgLayer, 'CardTypeError') && byName(systemMsgLayer, 'CardTypeError').activeInHierarchy),
            no_bigger_card: Boolean(systemMsgLayer && byName(systemMsgLayer, 'NoBiggerCard') && byName(systemMsgLayer, 'NoBiggerCard').activeInHierarchy),
            cant_lock: Boolean(systemMsgLayer && byName(systemMsgLayer, 'CantLock') && byName(systemMsgLayer, 'CantLock').activeInHierarchy),
          };

          const visibleShowCardSets = collectVisibleShowCardSets(showCardLayer);
          const visibleTableCards = visibleShowCardSets.flatMap((set) => set.cards);

          let turn = 'unknown';
          if (myGameClock && myGameClock.activeInHierarchy) {
            turn = 'self';
          } else if (enemyHourglass && enemyHourglass.activeInHierarchy) {
            const center = worldCenter(enemyHourglass);
            if (center) {
              if (center.x < 500) turn = 'left';
              else if (center.x > 1200) turn = 'right';
              else turn = 'top';
            }
          } else if (hintArrowLayer && hintArrowLayer.activeInHierarchy && hintArrow && hintArrow.activeInHierarchy) {
            const center = worldCenter(hintArrow);
            if (center) {
              if (center.y > 850) turn = 'top';
              else if (center.x < 500) turn = 'left';
              else if (center.x > 1200) turn = 'right';
              else turn = 'self';
            }
          } else {
            const highlightedEnemy = enemyProfiles.find((profile) => profile.highlighted && profile.center);
            if (highlightedEnemy) {
              if (highlightedEnemy.center.x < 500) turn = 'left';
              else if (highlightedEnemy.center.x > 1200) turn = 'right';
              else turn = 'top';
            }
          }

          return {
            scene: scene.name || null,
            my_hand_count: myCards.length,
            my_cards: myCards,
            my_selected_count: myCards.filter((card) => card.selected).length,
            my_playable_indexes: myCards
              .map((card, index) => ({ card, index }))
              .filter((entry) => !entry.card.disabled)
              .map((entry) => entry.index),
            enemy_profiles: enemyProfiles,
            turn,
            action_buttons: {
              pass: actionButton('PassBtn'),
              play: actionButton('PlayBtn'),
              cancel: actionButton('CancelBtn'),
            },
            card_type_buttons: cardTypeButtons,
            active_type_candidates: activeTypeCandidates,
            current_required_type: activeTypeCandidates.length === 1 ? activeTypeCandidates[0].key : null,
            visible_show_card_sets: visibleShowCardSets,
            visible_table_cards: visibleTableCards,
            visible_table_card_count: visibleTableCards.length,
            system_messages: systemMessages,
            change_three_active: Boolean(
              (changeCardLayer && changeCardLayer.activeInHierarchy) ||
              (changeConfirm && changeConfirm.activeInHierarchy) ||
              (changeClock && changeClock.activeInHierarchy)
            ),
            change_confirm: {
              active: Boolean(changeConfirm && changeConfirm.activeInHierarchy),
              center: worldCenter(changeConfirm),
            },
            change_clock_active: Boolean(changeClock && changeClock.activeInHierarchy),
            my_clock_active: Boolean(myGameClock && myGameClock.activeInHierarchy),
            enemy_hourglass_active: Boolean(enemyHourglass && enemyHourglass.activeInHierarchy),
            enemy_hourglass_center: worldCenter(enemyHourglass),
            hint_arrow_active: Boolean(hintArrowLayer && hintArrowLayer.activeInHierarchy && hintArrow && hintArrow.activeInHierarchy),
            hint_arrow_center: worldCenter(hintArrow),
            game_type: labelText(gameType),
            round: labelText(round),
          };
        }
        """
    )


async def click_design_point(page: Page, design_x: float, design_y: float) -> dict[str, float]:
    return await click_canvas_design_position(page, design_x, design_y)


async def ws_send_raw(page: Page, message: str) -> bool:
    return await page.evaluate(
        """(msg) => {
          const ws = window.__big2GameWebSocket;
          if (!ws || ws.readyState !== 1) return false;
          ws.send(msg);
          return true;
        }""",
        message,
    )
