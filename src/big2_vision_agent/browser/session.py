from __future__ import annotations

import json
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Error, Page, Playwright, async_playwright

from big2_vision_agent.config import Settings

NETWORK_HOOK_SCRIPT = r"""
(() => {
  if (window.__big2NetworkHookInstalled) {
    return;
  }
  window.__big2NetworkHookInstalled = true;
  window.__big2NetworkLog = [];
  window.__big2NetworkSeq = 0;

  function safeStringify(value) {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return String(value);
    }
  }

  function serializePayload(payload) {
    if (payload == null) {
      return null;
    }
    if (typeof payload === 'string') {
      return payload;
    }
    if (payload instanceof ArrayBuffer) {
      return `ArrayBuffer(${payload.byteLength})`;
    }
    if (typeof Blob !== 'undefined' && payload instanceof Blob) {
      return `Blob(${payload.size})`;
    }
    if (typeof payload === 'object') {
      return safeStringify(payload);
    }
    return String(payload);
  }

  function pushEntry(entry) {
    try {
      window.__big2NetworkLog.push({
        seq: ++window.__big2NetworkSeq,
        ts: Date.now(),
        ...entry,
      });
      if (window.__big2NetworkLog.length > 5000) {
        window.__big2NetworkLog.splice(0, window.__big2NetworkLog.length - 5000);
      }
    } catch (error) {}
  }

  const NativeWebSocket = window.WebSocket;
  if (NativeWebSocket) {
    window.WebSocket = function Big2HookedWebSocket(...args) {
      const ws = new NativeWebSocket(...args);
      const url = args[0];
      window.__big2GameWebSocket = ws;
      pushEntry({ kind: 'ws_open', url: String(url) });

      const nativeSend = ws.send;
      ws.send = function hookedSend(data) {
        pushEntry({
          kind: 'ws_send',
          url: String(url),
          payload: serializePayload(data),
        });
        return nativeSend.call(this, data);
      };

      ws.addEventListener('message', (event) => {
        pushEntry({
          kind: 'ws_message',
          url: String(url),
          payload: serializePayload(event.data),
        });
      });
      ws.addEventListener('close', (event) => {
        pushEntry({
          kind: 'ws_close',
          url: String(url),
          code: event.code,
          reason: event.reason,
          was_clean: event.wasClean,
        });
      });
      ws.addEventListener('error', () => {
        pushEntry({ kind: 'ws_error', url: String(url) });
      });
      return ws;
    };
    window.WebSocket.prototype = NativeWebSocket.prototype;
    Object.setPrototypeOf(window.WebSocket, NativeWebSocket);
  }

  const nativeFetch = window.fetch;
  if (nativeFetch) {
    window.fetch = async function hookedFetch(input, init) {
      const method = (init && init.method) || 'GET';
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      pushEntry({
        kind: 'fetch_request',
        method,
        url: String(url),
        body: serializePayload(init && init.body),
      });
      const response = await nativeFetch.apply(this, arguments);
      pushEntry({
        kind: 'fetch_response',
        method,
        url: String(url),
        status: response.status,
        ok: response.ok,
      });
      return response;
    };
  }

  const xhrProto = window.XMLHttpRequest && window.XMLHttpRequest.prototype;
  if (xhrProto) {
    const nativeOpen = xhrProto.open;
    const nativeSend = xhrProto.send;

    xhrProto.open = function hookedOpen(method, url, ...rest) {
      this.__big2Method = method;
      this.__big2Url = url;
      return nativeOpen.call(this, method, url, ...rest);
    };

    xhrProto.send = function hookedSend(body) {
      pushEntry({
        kind: 'xhr_request',
        method: this.__big2Method || 'GET',
        url: String(this.__big2Url || ''),
        body: serializePayload(body),
      });
      this.addEventListener('loadend', () => {
        pushEntry({
          kind: 'xhr_response',
          method: this.__big2Method || 'GET',
          url: String(this.__big2Url || ''),
          status: this.status,
        });
      }, { once: true });
      return nativeSend.call(this, body);
    };
  }
})();
"""


class BrowserSession:
    def __init__(self, settings: Settings, record_video_dir: Path | None = None) -> None:
        self.settings = settings
        self.record_video_dir = record_video_dir
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self.context: BrowserContext | None = None

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
        launch_kwargs = {
            "user_data_dir": str(self.settings.profile_dir),
            "headless": self.settings.headless,
            "viewport": {"width": 1440, "height": 1080},
            "args": ["--mute-audio"],
        }
        if self.record_video_dir is not None:
            self.record_video_dir.mkdir(parents=True, exist_ok=True)
            launch_kwargs["record_video_dir"] = str(self.record_video_dir)
            launch_kwargs["record_video_size"] = {"width": 1440, "height": 1080}
        try:
            self.context = await self._playwright.chromium.launch_persistent_context(
                **launch_kwargs,
            )
        except Error as error:
            if "ProcessSingleton" not in str(error):
                raise
            browser_kwargs = {
                "headless": self.settings.headless,
                "args": ["--mute-audio"],
            }
            self._browser = await self._playwright.chromium.launch(**browser_kwargs)
            context_kwargs = {
                "viewport": {"width": 1440, "height": 1080},
            }
            if self.settings.state_path.exists():
                context_kwargs["storage_state"] = str(self.settings.state_path)
            if self.record_video_dir is not None:
                self.record_video_dir.mkdir(parents=True, exist_ok=True)
                context_kwargs["record_video_dir"] = str(self.record_video_dir)
                context_kwargs["record_video_size"] = {"width": 1440, "height": 1080}
            self.context = await self._browser.new_context(**context_kwargs)
        self.context.set_default_timeout(self.settings.timeout_ms)
        await self.context.add_init_script(NETWORK_HOOK_SCRIPT)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Ctrl+C 或其他中斷可能導致瀏覽器連線已關閉，
        # 清理時忽略 Exception（TargetClosedError / ConnectionClosed 等）。
        try:
            if self.context is not None:
                await self.context.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:
            pass

    async def new_page(self) -> Page:
        if self.context is None:
            raise RuntimeError("Browser context is not ready.")
        page = await self.context.new_page()
        page.set_default_timeout(self.settings.timeout_ms)
        return page

    async def goto_home(self) -> Page:
        page = await self.new_page()
        await page.goto(self.settings.target_url, wait_until="domcontentloaded")
        await self._ensure_home_authenticated(page)
        if not await self._home_has_start_button(page):
            logged_in = await self.auto_facebook_login(page)
            if logged_in:
                await self.save_storage_state()
        return page

    async def goto_game(self) -> Page:
        page = await self.goto_home()
        page = await self._open_game_from_home(page)
        verified = await self._wait_for_real_game_page(page, timeout_ms=15000)
        if verified is not None:
            return verified
        return self._latest_live_page(preferred=page)

    async def goto_target(self) -> Page:
        return await self.goto_game()

    async def save_storage_state(self) -> Path:
        if self.context is None:
            raise RuntimeError("Browser context is not ready.")
        self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
        self.settings.state_path.parent.mkdir(parents=True, exist_ok=True)
        await self.context.storage_state(path=str(self.settings.state_path))
        return self.settings.state_path

    async def _open_game_from_home(self, page: Page) -> Page:
        if self.context is None:
            return page

        selectors = [
            'a.btn-start[onclick*="into_game"]',
            'a[onclick*="into_game(0,1,1)"]',
            'a[onclick*="into_game"]',
        ]
        start_link = None
        for selector in selectors:
            locator = page.locator(selector)
            if await locator.count() > 0:
                start_link = locator.first
                break
        if start_link is None:
            locator = page.get_by_role("link", name="開始遊戲")
            if await locator.count() == 0:
                locator = page.get_by_text("開始遊戲")
            if await locator.count() > 0:
                start_link = locator.first
        if start_link is None:
            if await page.locator('a.btn-start').count() > 0:
                start_link = page.locator('a.btn-start').first
            else:
                return page

        existing_pages = set(self.context.pages)
        try:
            await start_link.click(force=True)
        except Exception:
            await page.evaluate(
                """() => {
                    const button = document.querySelector('a.btn-start[onclick*="into_game"]');
                    if (button) {
                      button.click();
                      return;
                    }
                    if (typeof window.into_game === 'function') {
                      window.into_game(0, 1, 1);
                    }
                }"""
            )
        await page.wait_for_timeout(1500)

        new_pages = [candidate for candidate in self.context.pages if candidate not in existing_pages]
        if new_pages:
            game_page = await self._wait_for_real_game_page(new_pages[-1])
            if game_page is not None:
                return game_page

        if page.url.endswith("/bigtwo/#") or page.url.endswith("/bigtwo/"):
            try:
                await page.evaluate(
                    """() => {
                        if (typeof window.into_game === 'function') {
                          window.into_game(0, 1, 1);
                        }
                    }"""
                )
            except Exception:
                pass
            await page.wait_for_timeout(1500)

            new_pages = [candidate for candidate in self.context.pages if candidate not in existing_pages]
            if new_pages:
                game_page = await self._wait_for_real_game_page(new_pages[-1])
                if game_page is not None:
                    return game_page

        for _ in range(20):
            game_page = await self._wait_for_real_game_page(page, timeout_ms=500)
            if game_page is not None:
                return game_page
            page = self._latest_live_page(preferred=page)
            await page.wait_for_timeout(500)
        return self._latest_live_page(preferred=page)

    async def _wait_for_real_game_page(self, page: Page, timeout_ms: int | None = None) -> Page | None:
        if self.context is None:
            return page
        deadline = None
        if timeout_ms is not None:
            import asyncio
            deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)

        while True:
            candidate = self._latest_live_page(preferred=page)
            try:
                await candidate.wait_for_load_state("domcontentloaded")
            except Exception:
                pass
            if await self._is_real_game_page(candidate):
                return candidate
            if deadline is not None:
                import asyncio
                if asyncio.get_running_loop().time() >= deadline:
                    return None
            await candidate.wait_for_timeout(300)

    def _latest_live_page(self, preferred: Page | None = None) -> Page:
        if self.context is None:
            if preferred is None:
                raise RuntimeError("Browser context is not ready.")
            return preferred
        candidates = [page for page in self.context.pages if not page.is_closed()]
        if preferred is not None:
            try:
                if not preferred.is_closed():
                    candidates = [page for page in candidates if page != preferred] + [preferred]
            except Exception:
                pass
        return candidates[-1] if candidates else preferred

    async def _is_real_game_page(self, page: Page) -> bool:
        try:
            url = page.url
        except Exception:
            return False
        if "gamesofa.com/bigtwo/html5" not in url and "lobby.php" not in url:
            return False
        try:
            return await page.evaluate(
                """
                () => {
                  const hasCanvas = Boolean(document.querySelector('canvas'));
                  const hasScene = Boolean(
                    window.cc &&
                    window.cc.director &&
                    window.cc.director.getScene &&
                    window.cc.director.getScene()
                  );
                  return hasCanvas || hasScene;
                }
                """
            )
        except Exception:
            return False

    async def _ensure_home_authenticated(self, page: Page) -> None:
        if self.context is None:
            return
        if await self._home_has_start_button(page):
            return
        if not self.settings.state_path.exists():
            return
        try:
            storage = json.loads(self.settings.state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        cookies = storage.get("cookies") or []
        if cookies:
            try:
                await self.context.add_cookies(cookies)
            except Exception:
                pass

        origins = storage.get("origins") or []
        for origin_entry in origins:
            origin = origin_entry.get("origin")
            local_storage = origin_entry.get("localStorage") or []
            if not origin or not local_storage:
                continue
            try:
                await page.goto(origin, wait_until="domcontentloaded")
                await page.evaluate(
                    """(items) => {
                        for (const item of items) {
                            try {
                                window.localStorage.setItem(item.name, item.value);
                            } catch (error) {}
                        }
                    }""",
                    local_storage,
                )
            except Exception:
                continue

        try:
            await page.goto(self.settings.target_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)
        except Exception:
            return

    async def _home_has_start_button(self, page: Page) -> bool:
        for selector in (
            'a.btn-start',
            'a.btn-start[onclick*="into_game"]',
            'a[onclick*="into_game(0,1,1)"]',
            'a[onclick*="into_game"]',
        ):
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def auto_facebook_login(self, page: Page, timeout_ms: int = 30000) -> bool:
        """Click through the Facebook OAuth flow automatically.

        Flow:
          1. Navigate to the Gamesofa login page (?op=login_all).
          2. Click the Facebook login button (a[onclick*="fbLogin"]).
          3. FB SDK opens a popup → click 'Continue as [Name]' inside it.
          4. Popup closes, Gamesofa session is established.
          5. Navigate back to home and verify the start button appears.

        Assumes the Facebook session is already active in the browser profile.
        """
        import sys

        def _log(msg: str) -> None:
            print(f"[FB-login] {msg}", flush=True)

        # Step 1: go to login page where the FB button lives
        login_url = "https://www.gamesofa.com/bigtwo/?op=login_all"
        _log(f"Navigating to {login_url}")
        try:
            await page.goto(login_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
        except Exception as exc:
            _log(f"goto failed: {exc}")
            return False

        # Step 2: find the Facebook button
        fb_button = page.locator('a[onclick*="fbLogin"]').first
        count = await fb_button.count()
        _log(f"FB button count={count}")
        if count == 0:
            _log("FB login button not found on page; cannot auto-login")
            return False

        _log("Clicking FB login button and waiting for popup…")
        popup: Page | None = None

        # Listen for any new page that the context creates — covers both
        # synchronous window.open() and async calls from the FB JS SDK.
        import asyncio as _asyncio
        _popup_future: "_asyncio.Future[Page]" = _asyncio.get_running_loop().create_future()

        def _on_page(new_page: Page) -> None:
            if not _popup_future.done():
                _popup_future.set_result(new_page)

        if self.context is not None:
            self.context.on("page", _on_page)

        try:
            await fb_button.click()
        except Exception as exc:
            _log(f"FB button click error (non-fatal): {exc}")

        # Wait up to 12 s for the popup to appear
        try:
            popup = await _asyncio.wait_for(_popup_future, timeout=12)
            _log(f"Popup detected: {popup.url!r}")
        except _asyncio.TimeoutError:
            _log("No popup detected within 12 s")

        if self.context is not None:
            self.context.remove_listener("page", _on_page)

        if popup is not None:
            try:
                await popup.wait_for_load_state("domcontentloaded", timeout=10000)
                _log(f"Popup loaded: {popup.url!r}")
            except Exception as exc:
                _log(f"Popup load_state timeout (non-fatal): {exc}")

            # Save a debug screenshot so the user can see what we're looking at
            try:
                debug_path = self.settings.state_path.parent / "fb_popup_debug.png"
                await popup.screenshot(path=str(debug_path))
                _log(f"Saved popup screenshot → {debug_path}")
            except Exception:
                pass

            await self._click_facebook_continue(popup, timeout_ms, _log)

            # Wait for popup to close (either from our click or auto-close)
            try:
                await popup.wait_for_event("close", timeout=min(timeout_ms, 15000))
                _log("Popup closed")
            except Exception:
                _log("Popup did not close on its own; continuing anyway")
                try:
                    await popup.close()
                except Exception:
                    pass

        # Step 4: navigate back to home and verify
        _log(f"Navigating back to {self.settings.target_url}")
        try:
            await page.goto(self.settings.target_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2500)
        except Exception as exc:
            _log(f"Home navigation failed: {exc}")

        result = await self._home_has_start_button(page)
        _log(f"Auto-login result: {'success ✓' if result else 'failed ✗'}")
        return result

    async def _click_facebook_continue(self, page: Page, timeout_ms: int, _log=None) -> None:
        """Click the 'Continue as …' or equivalent confirmation button on the FB consent page."""
        if _log is None:
            def _log(msg: str) -> None:  # noqa: E306
                pass

        # Wait for the page to settle before probing
        try:
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 8000))
        except Exception:
            pass

        _log(f"Popup URL after settle: {page.url!r}")

        # Known selectors in rough order of likelihood (tested against FB OAuth 2024–2025)
        selectors = [
            'button[name="__CONFIRM__"]',            # classic FB dialog confirm
            '[data-testid="royal_login_button"]',    # FB "Log in" / "Continue" CTA
            'div[aria-label^="Continue as"] div[role="button"]',
            'div[aria-label^="以"] div[role="button"]',
            '[role="button"]:has-text("Continue as")',
            '[role="button"]:has-text("繼續使用")',
            '[role="button"]:has-text("繼續")',
            'button:has-text("Continue")',
            'button:has-text("OK")',
            'input[type="submit"]',
        ]

        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible", timeout=3000)
                _log(f"Clicking selector: {selector!r}")
                await locator.click()
                return
            except Exception:
                continue

        # Last-resort: dump visible interactive elements and click the first one
        _log("No known selector matched; falling back to first visible button")
        try:
            all_btns = page.locator('button, input[type="submit"], [role="button"]')
            total = await all_btns.count()
            _log(f"Found {total} button-like elements")
            for i in range(total):
                btn = all_btns.nth(i)
                try:
                    if await btn.is_visible():
                        label = await btn.inner_text()
                        _log(f"Clicking fallback button [{i}]: {label!r}")
                        await btn.click()
                        return
                except Exception:
                    continue
        except Exception as exc:
            _log(f"Fallback button scan failed: {exc}")
