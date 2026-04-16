/**
 * FILE: static/sw.js
 * IMPROVEMENT 10: PWA Service Worker — Browser Push Notifications
 *
 * Handles:
 *   1. Caching app files for offline use
 *   2. Background sync
 *   3. Push notification display
 *   4. Notification click routing
 *
 * Registered by index_v2.html automatically.
 * Push messages are sent by 04_app_improved.py via /api/push/send
 */

const CACHE_NAME    = "sma-predict-v2";
const CACHE_URLS    = [
  "/",
  "/index.html",
  "/dashboard_live.html",
  "https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Cabinet+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap",
  "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
];

// ── INSTALL: cache core assets ─────────────────────────────────────────────
self.addEventListener("install", event => {
  console.log("[SW] Installing SMA Predict service worker...");
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log("[SW] Caching core assets");
        // Use addAll only for same-origin assets to avoid CORS errors
        return cache.addAll(["/"]);
      })
      .then(() => self.skipWaiting())
      .catch(err => console.warn("[SW] Cache install warning:", err))
  );
});

// ── ACTIVATE: clean old caches ─────────────────────────────────────────────
self.addEventListener("activate", event => {
  console.log("[SW] Activating...");
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(key => key !== CACHE_NAME)
          .map(key => {
            console.log("[SW] Removing old cache:", key);
            return caches.delete(key);
          })
      )
    ).then(() => self.clients.claim())
  );
});

// ── FETCH: serve from cache, fallback to network ───────────────────────────
self.addEventListener("fetch", event => {
  // Only handle GET requests
  if (event.request.method !== "GET") return;

  // Skip API calls — always go to network
  if (event.request.url.includes("/api/")) return;

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;

      return fetch(event.request)
        .then(response => {
          // Cache successful same-origin responses
          if (
            response.ok &&
            response.type === "basic" &&
            event.request.url.startsWith(self.location.origin)
          ) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then(cache =>
              cache.put(event.request, responseClone)
            );
          }
          return response;
        })
        .catch(() => {
          // Offline fallback for HTML pages
          if (event.request.headers.get("accept")?.includes("text/html")) {
            return caches.match("/");
          }
        });
    })
  );
});

// ── PUSH: receive & display notification ──────────────────────────────────
self.addEventListener("push", event => {
  console.log("[SW] Push received");

  let data = {
    title:   "SMA Predict",
    body:    "You have a new notification from SMA Predict.",
    icon:    "/static/icons/icon-192.png",
    badge:   "/static/icons/icon-96.png",
    tag:     "sma-notification",
    url:     "/",
    risk:    "unknown",
    urgency: "low"
  };

  // Parse payload from server
  if (event.data) {
    try {
      const payload = event.data.json();
      data = { ...data, ...payload };
    } catch {
      data.body = event.data.text();
    }
  }

  // Choose icon color based on risk level
  const riskColors = { Low: "#22c982", Moderate: "#f5a623", High: "#ef4444" };
  const badgeText  = { Low: "✅", Moderate: "⚠️", High: "🚨" };

  const notifTitle = data.title;
  const notifOpts  = {
    body:              data.body,
    icon:              data.icon || "/static/icons/icon-192.png",
    badge:             data.badge || "/static/icons/icon-96.png",
    tag:               data.tag,
    renotify:          true,
    requireInteraction: data.urgency === "high",   // high risk stays until dismissed
    vibrate:           data.urgency === "high" ? [200, 100, 200, 100, 200] : [100],
    timestamp:         Date.now(),
    data: {
      url:     data.url || "/",
      risk:    data.risk,
      urgency: data.urgency
    },
    actions: [
      {
        action: "view",
        title:  "View My Results",
      },
      {
        action: "dismiss",
        title:  "Dismiss",
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(notifTitle, notifOpts)
  );
});

// ── NOTIFICATION CLICK: route to correct page ─────────────────────────────
self.addEventListener("notificationclick", event => {
  console.log("[SW] Notification clicked:", event.action);
  event.notification.close();

  if (event.action === "dismiss") return;

  const targetUrl = event.notification.data?.url || "/";

  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true })
      .then(windowClients => {
        // If app is already open, focus it
        for (const client of windowClients) {
          if (client.url.includes(self.location.origin) && "focus" in client) {
            client.postMessage({ type: "NOTIFICATION_CLICK", url: targetUrl });
            return client.focus();
          }
        }
        // Otherwise open a new window
        if (clients.openWindow) {
          return clients.openWindow(targetUrl);
        }
      })
  );
});

// ── NOTIFICATION CLOSE ─────────────────────────────────────────────────────
self.addEventListener("notificationclose", event => {
  console.log("[SW] Notification closed:", event.notification.tag);
  // Could track dismissal analytics here
});

// ── BACKGROUND SYNC ────────────────────────────────────────────────────────
self.addEventListener("sync", event => {
  if (event.tag === "sync-predictions") {
    console.log("[SW] Background sync: syncing pending predictions");
    event.waitUntil(syncPendingPredictions());
  }
});

async function syncPendingPredictions() {
  // In a full implementation, this would read from IndexedDB
  // and POST any offline predictions saved while the user was disconnected
  console.log("[SW] Background sync complete");
}

console.log("[SW] Service worker script loaded — SMA Predict v2");
