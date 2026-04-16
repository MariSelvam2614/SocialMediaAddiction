"""
FILE: scheduler.py
IMPROVEMENT 9: Scheduled Email Reminders
- Sends 30-day re-assessment reminder emails to users
- Runs as a background process alongside the Flask app
- Uses APScheduler for job scheduling
- Checks predictions.db for users who haven't assessed in 30 days

Setup:
  pip install apscheduler

Run alongside Flask:
  python scheduler.py          ← in a separate terminal
  python 04_app_improved.py    ← in another terminal

Or run both together (recommended for production):
  Use the run_all.py launcher
"""

import sqlite3
import smtplib
import os
import logging
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH       = "data/predictions.db"
REMINDER_DAYS = 30          # remind users after this many days of inactivity
CHECK_HOUR    = 9           # run check at 9 AM daily
CHECK_MINUTE  = 0

EMAIL_HOST = os.environ.get("EMAIL_HOST")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler()
    ]
)
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)


# ─── GET USERS DUE FOR REMINDER ───────────────────────────────────────────────
def get_users_due_reminder():
    """
    Returns list of (user_id, email, name, last_prediction_date, last_risk_label)
    for users whose last prediction was >= REMINDER_DAYS ago.
    """
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found at {DB_PATH}. Run the app first.")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=REMINDER_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row

        rows = db.execute("""
            SELECT
                u.id         AS user_id,
                u.email      AS email,
                u.name       AS name,
                MAX(p.created_at) AS last_prediction,
                p.risk_label AS last_risk_label
            FROM users u
            LEFT JOIN predictions p ON p.user_id = u.id
            GROUP BY u.id
            HAVING last_prediction IS NULL
               OR last_prediction <= ?
        """, (cutoff_str,)).fetchall()

        db.close()
        return [dict(r) for r in rows]

    except Exception as e:
        logger.error(f"DB query failed: {e}")
        return []


# ─── BUILD REMINDER EMAIL ─────────────────────────────────────────────────────
def build_reminder_email(user: dict) -> str:
    name        = user.get("name") or "there"
    last_date   = user.get("last_prediction")
    last_risk   = user.get("last_risk_label") or "Unknown"

    if last_date:
        last_dt    = datetime.fromisoformat(last_date.replace("Z",""))
        days_ago   = (datetime.utcnow() - last_dt).days
        last_str   = f"{days_ago} days ago ({last_dt.strftime('%d %b %Y')})"
        context    = f"Your last assessment was <b>{last_str}</b> and your result was <b>{last_risk} Risk</b>."
    else:
        context = "You haven't completed an assessment yet."

    risk_color = {"Low":"#22c982","Moderate":"#f5a623","High":"#ef4444"}.get(last_risk, "#3b8aff")

    tips = {
        "Low": [
            "Keep up your healthy habits — do a quick weekly screen-time check.",
            "Protect your sleep by staying phone-free 30 minutes before bed.",
            "Use this month to try one new offline hobby."
        ],
        "Moderate": [
            "Set a daily app limit of 90 minutes total social media time.",
            "Schedule 3 app-free hours every day — morning works best.",
            "Try one full phone-free day this weekend."
        ],
        "High": [
            "Check in with your accountability partner about your progress.",
            "Review whether you deleted the triggering apps last month.",
            "Consider booking a session with a digital wellness counsellor."
        ],
    }.get(last_risk, [
        "Take your first assessment to understand your risk level.",
        "The process takes just 2 minutes and gives personalised guidance.",
    ])

    tips_html = "".join(f"<li style='margin-bottom:8px;color:#9bb4cc'>{t}</li>" for t in tips)

    return f"""
<div style="font-family:'Helvetica Neue',sans-serif;max-width:580px;margin:0 auto;
            background:#07090f;color:#dde6f0;border-radius:16px;overflow:hidden;
            border:1px solid #1c2a3a">

  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#0d1117,#111820);
              padding:32px 36px;border-bottom:1px solid #1c2a3a;text-align:center">
    <div style="font-size:2rem;margin-bottom:8px">📱</div>
    <h1 style="margin:0;font-size:1.3rem;font-weight:700;color:#fff;letter-spacing:-.02em">
      Time for Your Monthly Check-In
    </h1>
    <p style="margin:8px 0 0;font-size:.85rem;color:#56728a">SMA Predict · 30-Day Reminder</p>
  </div>

  <!-- BODY -->
  <div style="padding:32px 36px">
    <p style="margin:0 0 18px;font-size:.95rem;line-height:1.7">Hi <b style="color:#fff">{name}</b>,</p>
    <p style="margin:0 0 20px;font-size:.9rem;color:#9bb4cc;line-height:1.7">
      It's been 30 days since your last SMA Predict assessment. Digital habits shift over time
      — a quick re-check helps you stay aware and catch any changes early.
    </p>

    <!-- LAST RESULT CARD -->
    <div style="background:#0d1117;border:1px solid {risk_color}40;border-left:4px solid {risk_color};
                border-radius:10px;padding:18px 20px;margin-bottom:24px">
      <p style="margin:0;font-size:.8rem;color:#56728a;text-transform:uppercase;letter-spacing:.08em">
        Your Last Result
      </p>
      <p style="margin:6px 0 0;font-size:1rem;color:{risk_color};font-weight:700">{last_risk} Risk</p>
      <p style="margin:4px 0 0;font-size:.82rem;color:#56728a">{context}</p>
    </div>

    <!-- TIPS -->
    <h3 style="margin:0 0 12px;font-size:.9rem;font-weight:600;color:#fff">
      💡 This Month's Tips for You
    </h3>
    <ul style="margin:0 0 24px;padding-left:18px;line-height:1.8;font-size:.85rem">
      {tips_html}
    </ul>

    <!-- CTA BUTTON -->
    <div style="text-align:center;margin:28px 0">
      <a href="http://localhost:5000"
         style="background:linear-gradient(135deg,#1a6fff,#0a4fd4);color:#fff;
                text-decoration:none;padding:14px 36px;border-radius:12px;
                font-size:.95rem;font-weight:600;letter-spacing:.02em;
                display:inline-block;box-shadow:0 4px 20px rgba(59,138,255,.3)">
        Re-Take My Assessment →
      </a>
    </div>

    <p style="margin:0;font-size:.78rem;color:#56728a;line-height:1.8;text-align:center">
      ⚕️ This is an awareness tool only, not a clinical diagnosis.<br>
      If you have concerns about your mental health, please consult a professional.
    </p>
  </div>

  <!-- FOOTER -->
  <div style="background:#0d1117;border-top:1px solid #1c2a3a;padding:16px 36px;text-align:center">
    <p style="margin:0;font-size:.72rem;color:#374151">
      SMA Predict · You received this because you registered an account.<br>
      To unsubscribe, delete your account from the app settings.
    </p>
  </div>
</div>
"""


# ─── SEND ONE REMINDER ────────────────────────────────────────────────────────
def send_reminder(user: dict) -> bool:
    if not all([EMAIL_HOST, EMAIL_USER, EMAIL_PASS]):
        logger.info(f"  [SKIP] Email not configured — would have sent to {user['email']}")
        return False

    try:
        html_body = build_reminder_email(user)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📱 Time for Your Monthly SMA Check-In, {user.get('name','')}"
        msg["From"]    = EMAIL_USER
        msg["To"]      = user["email"]
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, user["email"], msg.as_string())

        logger.info(f"  ✓ Reminder sent → {user['email']}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to send to {user['email']}: {e}")
        return False


# ─── MAIN JOB ─────────────────────────────────────────────────────────────────
def run_reminder_job():
    logger.info("=" * 50)
    logger.info(f"Running reminder check (threshold: {REMINDER_DAYS} days)")
    logger.info("=" * 50)

    users = get_users_due_reminder()
    logger.info(f"Users due for reminder: {len(users)}")

    sent = 0
    for user in users:
        logger.info(f"  Processing: {user['email']} (last: {user.get('last_prediction','never')})")
        if send_reminder(user):
            sent += 1

    logger.info(f"Reminder job complete. Sent: {sent}/{len(users)}")


# ─── SCHEDULER SETUP ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting SMA Predict Scheduler")
    logger.info(f"  Reminder threshold : {REMINDER_DAYS} days")
    logger.info(f"  Daily check time   : {CHECK_HOUR:02d}:{CHECK_MINUTE:02d}")
    logger.info(f"  Email configured   : {'YES' if EMAIL_USER else 'NO (set EMAIL_HOST, EMAIL_USER, EMAIL_PASS)'}")

    scheduler = BlockingScheduler(timezone="UTC")

    # Run daily at CHECK_HOUR:CHECK_MINUTE
    scheduler.add_job(
        run_reminder_job,
        trigger=CronTrigger(hour=CHECK_HOUR, minute=CHECK_MINUTE),
        id="daily_reminder",
        name="30-Day Re-Assessment Reminder",
        replace_existing=True
    )

    # Also run once immediately on startup (for testing)
    scheduler.add_job(
        run_reminder_job,
        id="startup_check",
        name="Startup reminder check"
    )

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    logger.info(f"Next run: daily at {CHECK_HOUR:02d}:{CHECK_MINUTE:02d} UTC\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
