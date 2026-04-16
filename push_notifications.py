"""
FILE: push_notifications.py
IMPROVEMENT 10: Browser Push Notifications (PWA)

Adds push notification support to the Flask app:
  - Stores browser push subscriptions in the database
  - Sends web push messages after predictions
  - Sends reminders via push (works alongside scheduler.py)

Setup:
  pip install pywebpush

Generate VAPID keys (run ONCE, save the output):
  python push_notifications.py --generate-keys

Add keys to your environment:
  export VAPID_PRIVATE_KEY=<your_private_key>
  export VAPID_PUBLIC_KEY=<your_public_key>
  export VAPID_CLAIMS_EMAIL=you@example.com

Then import and register with Flask:
  from push_notifications import register_push_routes
  register_push_routes(app)
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

VAPID_PRIVATE_KEY   = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY    = os.environ.get("VAPID_PUBLIC_KEY",  "")
VAPID_CLAIMS_EMAIL  = os.environ.get("VAPID_CLAIMS_EMAIL","admin@smapredict.app")
DB_PATH             = "data/predictions.db"


# ─── DATABASE HELPERS ─────────────────────────────────────────────────────────
def init_push_table():
    """Add push_subscriptions table to the database."""
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute("""
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                endpoint    TEXT NOT NULL UNIQUE,
                p256dh      TEXT NOT NULL,
                auth        TEXT NOT NULL,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        db.commit()
        db.close()
        logger.info("✓ push_subscriptions table ready")
    except Exception as e:
        logger.warning(f"Push table init: {e}")


def save_subscription(user_id: str, subscription: dict):
    """Save a browser push subscription to the database."""
    try:
        db = sqlite3.connect(DB_PATH)
        keys = subscription.get("keys", {})
        db.execute("""
            INSERT OR REPLACE INTO push_subscriptions
                (user_id, endpoint, p256dh, auth)
            VALUES (?, ?, ?, ?)
        """, (
            user_id,
            subscription["endpoint"],
            keys.get("p256dh", ""),
            keys.get("auth", "")
        ))
        db.commit()
        db.close()
        return True
    except Exception as e:
        logger.error(f"Save subscription failed: {e}")
        return False


def get_subscriptions(user_id: str = None) -> list:
    """Get push subscriptions — all users or a specific user."""
    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        if user_id:
            rows = db.execute(
                "SELECT * FROM push_subscriptions WHERE user_id=?", (user_id,)
            ).fetchall()
        else:
            rows = db.execute("SELECT * FROM push_subscriptions").fetchall()
        db.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Get subscriptions failed: {e}")
        return []


def delete_subscription(endpoint: str):
    """Remove an expired or invalid subscription."""
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute("DELETE FROM push_subscriptions WHERE endpoint=?", (endpoint,))
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Delete subscription failed: {e}")


# ─── SEND PUSH MESSAGE ────────────────────────────────────────────────────────
def send_push(subscription_row: dict, payload: dict) -> bool:
    """
    Send a Web Push message to one subscription.
    Returns True on success, False on failure.
    """
    if not VAPID_PRIVATE_KEY:
        logger.info("[PUSH] VAPID keys not configured — skipping push send")
        logger.info(f"[PUSH] Would have sent: {payload.get('title','')}")
        return False

    try:
        from pywebpush import webpush, WebPushException

        webpush(
            subscription_info={
                "endpoint": subscription_row["endpoint"],
                "keys": {
                    "p256dh": subscription_row["p256dh"],
                    "auth":   subscription_row["auth"]
                }
            },
            data=json.dumps(payload),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={
                "sub": f"mailto:{VAPID_CLAIMS_EMAIL}",
            }
        )
        return True

    except Exception as e:
        err_str = str(e)
        if "410" in err_str or "404" in err_str:
            # Subscription expired — clean it up
            delete_subscription(subscription_row["endpoint"])
            logger.info(f"[PUSH] Removed expired subscription: {subscription_row['endpoint'][:40]}...")
        else:
            logger.error(f"[PUSH] Send failed: {e}")
        return False


def send_push_to_user(user_id: str, payload: dict) -> int:
    """Send push notification to all subscriptions for a user. Returns count sent."""
    subs = get_subscriptions(user_id)
    sent = 0
    for sub in subs:
        if send_push(sub, payload):
            sent += 1
    return sent


def send_push_to_all(payload: dict) -> int:
    """Broadcast push to all subscribed users. Returns count sent."""
    subs = get_subscriptions()
    sent = 0
    for sub in subs:
        if send_push(sub, payload):
            sent += 1
    return sent


# ─── NOTIFICATION PAYLOADS ────────────────────────────────────────────────────
def prediction_payload(risk_label: str, confidence: float, user_name: str = "") -> dict:
    """Build push payload for a completed prediction."""
    icons   = {"Low": "✅", "Moderate": "⚠️", "High": "🚨"}
    urgency = {"Low": "low", "Moderate": "medium", "High": "high"}
    bodies  = {
        "Low":      "Great news — your social media habits look healthy. See your personalised tips.",
        "Moderate": "Moderate risk detected. Your action plan is ready — check it now.",
        "High":     "High addiction risk found. Open your results for immediate recommendations."
    }
    return {
        "title":   f"{icons.get(risk_label,'📊')} Your SMA Result: {risk_label} Risk",
        "body":    bodies.get(risk_label, "Your assessment is complete."),
        "icon":    "/static/icons/icon-192.png",
        "badge":   "/static/icons/icon-96.png",
        "tag":     "sma-prediction",
        "url":     "/",
        "risk":    risk_label,
        "urgency": urgency.get(risk_label, "low"),
        "confidence": round(confidence * 100)
    }


def reminder_payload(user_name: str = "", last_risk: str = "") -> dict:
    """Build push payload for a 30-day reminder."""
    return {
        "title":   "📱 Monthly Check-In Reminder",
        "body":    f"It's been 30 days since your last assessment{f', {user_name}' if user_name else ''}. Re-assess in 2 minutes.",
        "icon":    "/static/icons/icon-192.png",
        "badge":   "/static/icons/icon-96.png",
        "tag":     "sma-reminder",
        "url":     "/",
        "risk":    last_risk,
        "urgency": "medium"
    }


# ─── FLASK ROUTE REGISTRATION ─────────────────────────────────────────────────
def register_push_routes(app):
    """
    Call this from 04_app_improved.py to add push routes:

        from push_notifications import register_push_routes
        register_push_routes(app)
    """
    from flask import request, jsonify, g
    init_push_table()

    @app.route("/api/push/vapid-public-key", methods=["GET"])
    def get_vapid_key():
        """Frontend calls this to get the VAPID public key for subscribing."""
        return jsonify({"public_key": VAPID_PUBLIC_KEY or "NOT_CONFIGURED"})

    @app.route("/api/push/subscribe", methods=["POST"])
    def push_subscribe():
        """Save a browser push subscription from the frontend."""
        # Require auth
        auth_header = request.headers.get("Authorization","")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401

        body = request.get_json(force=True, silent=True) or {}
        subscription = body.get("subscription")
        if not subscription or "endpoint" not in subscription:
            return jsonify({"error": "Invalid subscription object"}), 400

        # Decode user_id from token
        try:
            import jwt as pyjwt
            JWT_SECRET = os.environ.get("JWT_SECRET","sma-jwt-secret-change-in-prod")
            token   = auth_header.split(" ")[1]
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = payload["sub"]
        except Exception as e:
            return jsonify({"error": "Invalid token"}), 401

        ok = save_subscription(user_id, subscription)
        if ok:
            logger.info(f"Push subscription saved for user {user_id}")
            return jsonify({"message": "Subscribed successfully"}), 201
        return jsonify({"error": "Failed to save subscription"}), 500

    @app.route("/api/push/unsubscribe", methods=["POST"])
    def push_unsubscribe():
        """Remove a push subscription."""
        body     = request.get_json(force=True, silent=True) or {}
        endpoint = body.get("endpoint")
        if endpoint:
            delete_subscription(endpoint)
        return jsonify({"message": "Unsubscribed"}), 200

    @app.route("/api/push/test", methods=["POST"])
    def push_test():
        """Send a test push to the authenticated user."""
        auth_header = request.headers.get("Authorization","")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            import jwt as pyjwt
            JWT_SECRET = os.environ.get("JWT_SECRET","sma-jwt-secret-change-in-prod")
            token   = auth_header.split(" ")[1]
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = payload["sub"]
        except:
            return jsonify({"error": "Invalid token"}), 401

        test_payload = {
            "title": "🔔 Test Push Notification",
            "body":  "Push notifications are working correctly for SMA Predict!",
            "icon":  "/static/icons/icon-192.png",
            "tag":   "sma-test",
            "url":   "/"
        }
        sent = send_push_to_user(user_id, test_payload)
        return jsonify({"sent": sent, "message": f"Test push sent to {sent} subscription(s)"}), 200

    logger.info("✓ Push notification routes registered")


# ─── ICON GENERATOR ───────────────────────────────────────────────────────────
def generate_icons():
    """
    Generate placeholder PNG icons for the PWA manifest.
    Run once: python push_notifications.py --generate-icons

    For production, replace with proper branded icons.
    """
    import struct, zlib

    def make_png(size, bg_color=(11,15,26), fg_color=(0,217,180)):
        """Create a minimal valid PNG with a colored circle icon."""
        def write_chunk(chunk_type, data):
            chunk = chunk_type + data
            return (
                struct.pack(">I", len(data)) +
                chunk +
                struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            )

        cx = cy = size // 2
        r  = int(size * 0.38)

        raw_rows = []
        for y in range(size):
            row = [0]  # filter byte
            for x in range(size):
                dist = ((x-cx)**2 + (y-cy)**2) ** 0.5
                if dist <= r:
                    row.extend(fg_color)
                else:
                    row.extend(bg_color)
            raw_rows.append(bytes(row))

        compressed = zlib.compress(b"".join(raw_rows), 9)

        png  = b"\x89PNG\r\n\x1a\n"
        png += write_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        png += write_chunk(b"IDAT", compressed)
        png += write_chunk(b"IEND", b"")
        return png

    icon_dir = "static/icons"
    os.makedirs(icon_dir, exist_ok=True)

    sizes = [72, 96, 128, 144, 192, 512]
    for size in sizes:
        path = f"{icon_dir}/icon-{size}.png"
        with open(path, "wb") as f:
            f.write(make_png(size))
        print(f"  ✓ Generated {path}")

    print(f"\n  Icons saved to {icon_dir}/")
    print("  Replace with branded icons before deploying to production.")


# ─── VAPID KEY GENERATOR ──────────────────────────────────────────────────────
def generate_vapid_keys():
    """Generate VAPID key pair for push notifications."""
    try:
        from py_vapid import Vapid
        vapid = Vapid()
        vapid.generate_keys()
        private = vapid.private_pem().decode()
        public  = vapid.public_key.public_bytes(
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding","PublicFormat"]).Encoding.PEM,
            __import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding","PublicFormat"]).PublicFormat.SubjectPublicKeyInfo
        ).decode()
        print("\n  ✓ VAPID Keys Generated")
        print("  Add these to your environment:\n")
        print(f'  export VAPID_PRIVATE_KEY="{private.strip()}"')
        print(f'  export VAPID_PUBLIC_KEY="{public.strip()}"')
        print('  export VAPID_CLAIMS_EMAIL="your@email.com"')
    except ImportError:
        print("  Install: pip install py-vapid")
        print("  Then run: python push_notifications.py --generate-keys")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--generate-keys" in sys.argv:
        generate_vapid_keys()
    elif "--generate-icons" in sys.argv:
        generate_icons()
        print("\n  Also generating icons now...")
    else:
        print("Usage:")
        print("  python push_notifications.py --generate-keys    # Generate VAPID keys")
        print("  python push_notifications.py --generate-icons   # Generate PWA icons")
        print()
        print("  To use in Flask app, add to 04_app_improved.py:")
        print("    from push_notifications import register_push_routes")
        print("    register_push_routes(app)")
        print()
        generate_icons()
