"""
Milestone 1 – Data Collection & Preprocessing
Step 1: Synthetic Support Ticket Dataset Generator

Generates a realistic, labelled support-ticket corpus that covers:
  - 6 categories  (billing, technical, account, shipping, product, general)
  - 3 priority levels (low, medium, high)
  - ticket text + auto-generated resolution notes
  - basic metadata (ticket_id, created_at, source_channel, agent_id)

Output: data/raw/support_tickets.csv
"""

import random
import uuid
import csv
import os
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

# ─────────────────────────────────────────────
# 1. Template bank – (subject_templates, body_templates, resolution_templates)
#    Each entry is (text_template, category, priority_weight)
#    priority_weight: list[float] for [low, medium, high]
# ─────────────────────────────────────────────

CATEGORIES = {
    "billing": {
        "priority_weights": [0.3, 0.4, 0.3],
        "subjects": [
            "Incorrect charge on my invoice",
            "Double billing issue",
            "Refund not received after {days} days",
            "Cannot access invoice #{num}",
            "Unexpected subscription renewal charge",
            "Billing address update not reflected",
            "Discount code not applied to my bill",
            "Tax exemption certificate not processed",
        ],
        "bodies": [
            "I was charged ${amount} on {date} but this does not match my plan. My account email is {email}. Please review and correct.",
            "My account shows two identical charges of ${amount} on the same date. I need a refund for the duplicate.",
            "I requested a refund {days} days ago (ticket ref {ref}) but have not received it yet. Please expedite.",
            "I upgraded my plan but the old price is still showing on invoice #{num}. The new rate should be ${amount}/month.",
            "I cancelled my subscription on {date} but was still charged ${amount} for the next cycle.",
            "I updated my billing address last week but my latest invoice still shows the old address.",
        ],
        "resolutions": [
            "Verified the charge discrepancy. Issued a credit of ${amount} to the account. Customer notified.",
            "Confirmed duplicate charge. Refund of ${amount} initiated and will reflect in 3–5 business days.",
            "Located the refund request. Escalated to finance team. ETA updated and customer informed.",
            "Updated invoice to reflect correct pricing. Sent amended invoice to customer.",
            "Subscription cancellation confirmed. Refund processed for the extra billing cycle.",
        ],
    },

    "technical": {
        "priority_weights": [0.15, 0.35, 0.50],
        "subjects": [
            "Application crashes on startup",
            "Login page returns 500 error",
            "API rate limit exceeded unexpectedly",
            "Data export feature not working",
            "Dashboard loading very slowly",
            "Integration with {tool} broken after update",
            "Two-factor authentication not sending OTP",
            "File upload fails for files over {size}MB",
        ],
        "bodies": [
            "The app crashes every time I open it since the update on {date}. I am on {os} version {version}. Error log: {error}.",
            "I get a 500 Internal Server Error when trying to log in. This started {days} days ago. Browser: {browser}.",
            "Our API key is hitting rate limits even though we are well within the documented quota. Account ID: {num}.",
            "Trying to export data as CSV returns a blank file. Happens for date ranges over {days} days.",
            "The dashboard takes over {duration} seconds to load. This is affecting our team's productivity.",
            "Since the v{version} update, our {tool} integration no longer syncs data. Last successful sync: {date}.",
            "OTP for 2FA is not arriving via SMS or email. Tested with multiple numbers. Account: {email}.",
            "Uploads above {size}MB fail with a timeout error. We regularly upload large datasets.",
        ],
        "resolutions": [
            "Identified a bug in v{version} causing crashes on {os}. Hotfix deployed. Customer asked to update.",
            "Root cause: database connection pool exhaustion. Increased pool size. Monitored for 1 hour – stable.",
            "Found misconfiguration in rate-limit tier for this account. Corrected and documented.",
            "Fixed CSV export encoding issue in backend. Customer re-ran export successfully.",
            "Identified slow query in dashboard aggregation. Added index. Load time reduced to under 2s.",
            "Integration token expired silently after update. Guided customer to re-authenticate. Sync restored.",
            "OTP delivery issue traced to SMS provider outage. Switched to backup provider. OTP now working.",
            "Increased upload size limit for account plan. Customer confirmed large files uploading correctly.",
        ],
    },

    "account": {
        "priority_weights": [0.35, 0.40, 0.25],
        "subjects": [
            "Cannot reset my password",
            "Account locked after too many login attempts",
            "Need to update email address on file",
            "Request to merge two accounts",
            "Profile photo not updating",
            "Cannot delete my account",
            "Team member invite not received",
            "Change primary account owner",
        ],
        "bodies": [
            "The password reset email is not arriving at {email}. I have checked spam. Please help me reset manually.",
            "My account {email} is locked after failed login attempts. I am the legitimate owner. Please unlock.",
            "I need to change my account email from {email} to a new address. Please advise the process.",
            "I have two accounts ({email} and another) that I want to merge. Please guide me.",
            "I uploaded a new profile photo but the old one keeps showing even after clearing cache.",
            "I want to close my account permanently but the delete option is greyed out in settings.",
            "I sent a team invite to {email} {days} days ago but they say they never received it.",
            "I am leaving the company. Need to transfer account ownership to {email}.",
        ],
        "resolutions": [
            "Checked email delivery logs – reset email was going to spam folder. Resent and advised customer.",
            "Unlocked account after identity verification. Advised customer to use password manager.",
            "Sent email change verification link. Customer confirmed update completed.",
            "Accounts merged successfully. Data audit performed. Customer notified.",
            "Photo CDN cache cleared for account. Customer confirmed new photo is visible.",
            "Account deletion restriction due to active subscription. Customer cancelled subscription; account deleted.",
            "Found invite in spam filter configuration list. Whitelisted domain. Resent invite successfully.",
            "Ownership transfer process completed after verification of both parties.",
        ],
    },

    "shipping": {
        "priority_weights": [0.20, 0.45, 0.35],
        "subjects": [
            "Order #{num} not delivered after {days} days",
            "Received wrong item in my order",
            "Package arrived damaged",
            "Tracking number not updating",
            "Request to change delivery address",
            "Order shipped to wrong address",
            "Missing item from multi-item order",
            "Return shipment not picked up",
        ],
        "bodies": [
            "My order #{num} was supposed to arrive {days} days ago but I still haven't received it. Tracking: {ref}.",
            "I received a completely wrong item. I ordered {product} but received something else entirely. Order #{num}.",
            "My package arrived with significant damage. Contents are broken. Order #{num}. Photos attached.",
            "The tracking for order #{num} has been stuck on '{status}' for {days} days with no updates.",
            "I need to change the delivery address for order #{num} before it ships. New address: {address}.",
            "My order was delivered to an old address even though I updated my address before placing the order.",
            "My order #{num} should have contained {num2} items but only {num3} arrived. Missing: {product}.",
            "I scheduled a return pickup {days} days ago but the courier never showed up.",
        ],
        "resolutions": [
            "Investigated with carrier. Package was delayed at regional hub. Expedited delivery – arrived next day.",
            "Wrong item confirmed. Correct item reshipped with priority delivery. Return label sent for wrong item.",
            "Damage claim filed with carrier. Replacement order created and shipped. Photo evidence retained.",
            "Tracking system glitch identified. Actual package in transit. Customer updated with real ETA.",
            "Address updated in system before dispatch. Customer confirmed new address is correct.",
            "Investigated: address was cached from previous order. Refund issued; reshipment to correct address.",
            "Missing item located in warehouse. Shipped separately with tracking. Customer confirmed receipt.",
            "Rescheduled pickup with courier. Customer confirmed pickup completed.",
        ],
    },

    "product": {
        "priority_weights": [0.45, 0.40, 0.15],
        "subjects": [
            "Feature request: {feature}",
            "How do I use {feature}?",
            "Product documentation is unclear for {feature}",
            "Compatibility question about {tool}",
            "Suggestion to improve {feature}",
            "Is {feature} available on the mobile app?",
            "Difference between {plan1} and {plan2} plans",
            "Looking for a tutorial on {feature}",
        ],
        "bodies": [
            "It would be great if you could add {feature} to the product. This would save our team a lot of time.",
            "I can't figure out how to use {feature}. The documentation doesn't explain the steps clearly.",
            "The docs for {feature} are outdated – the screenshots don't match the current UI.",
            "We want to integrate with {tool}. Can you confirm compatibility with your API v{version}?",
            "The {feature} works but could be much better if you added {suggestion}.",
            "I can access {feature} on the web but it doesn't seem to exist in the mobile app. Any plans to add it?",
            "I'm trying to decide between {plan1} and {plan2}. What are the key differences for a team of {num} people?",
            "Do you have a video tutorial or step-by-step guide for setting up {feature}?",
        ],
        "resolutions": [
            "Feature request logged and passed to the product team. Customer subscribed to feature updates.",
            "Provided step-by-step guide for {feature} and shared link to updated documentation.",
            "Docs team notified. Updated documentation sent directly to customer.",
            "Confirmed compatibility. Shared integration guide and API credentials needed.",
            "Enhancement suggestion logged for next sprint review. Customer thanked for feedback.",
            "Confirmed {feature} is on the mobile roadmap for Q{quarter}. Customer added to beta list.",
            "Provided detailed plan comparison table. Customer decided to proceed with the higher tier.",
            "Shared tutorial video and written guide. Customer confirmed issue resolved.",
        ],
    },

    "general": {
        "priority_weights": [0.60, 0.30, 0.10],
        "subjects": [
            "General inquiry about your service",
            "How do I get started?",
            "What are your business hours?",
            "Looking for a partner/reseller program",
            "Accessibility features available?",
            "Is my data GDPR compliant?",
            "Looking for a case study or demo",
            "Request for enterprise pricing",
        ],
        "bodies": [
            "I came across your service online and want to know more about what you offer before signing up.",
            "I just created an account. Where should I start? Is there an onboarding guide?",
            "What are your customer support hours and response time commitments?",
            "We are interested in reselling your product. Do you have a partner programme?",
            "We need to confirm the platform meets WCAG 2.1 AA accessibility standards before purchasing.",
            "We are a company in the EU. Can you confirm your data storage and processing is GDPR compliant?",
            "Can you share a case study for a company similar to ours, or arrange a product demo?",
            "We have a team of {num} users. Can you provide custom/enterprise pricing?",
        ],
        "resolutions": [
            "Provided product overview and feature sheet. Offered to schedule a demo call.",
            "Sent onboarding guide and welcome email with quick-start video.",
            "Shared SLA document outlining support hours and response time commitments by plan tier.",
            "Forwarded to partnerships team. Partner program brochure sent.",
            "Shared WCAG 2.1 AA compliance documentation. Offered accessibility audit report.",
            "Sent DPA and GDPR compliance documentation. Offered to connect with our DPO.",
            "Scheduled 30-minute demo. Sent relevant case study PDF post-call.",
            "Escalated to enterprise sales team. Custom proposal sent within 24 hours.",
        ],
    },
}

CHANNELS = ["email", "web_form", "chat", "phone_callback", "social_media"]
STATUSES  = ["open", "in_progress", "resolved", "closed", "escalated"]
AGENTS    = [f"AGT-{str(i).zfill(3)}" for i in range(1, 21)]

PRIORITIES = ["low", "medium", "high"]


def weighted_priority(weights):
    return random.choices(PRIORITIES, weights=weights, k=1)[0]


def fill(template):
    """Replace placeholder tokens with random fake data."""
    replacements = {
        "{amount}":    str(round(random.uniform(9.99, 499.99), 2)),
        "{date}":      fake.date_between(start_date="-60d", end_date="today").strftime("%Y-%m-%d"),
        "{days}":      str(random.randint(2, 30)),
        "{num}":       str(random.randint(1000, 99999)),
        "{num2}":      str(random.randint(2, 6)),
        "{num3}":      str(random.randint(1, 5)),
        "{ref}":       fake.bothify("TK-######"),
        "{email}":     fake.email(),
        "{error}":     fake.bothify("ERR-????-####"),
        "{os}":        random.choice(["Windows 11", "macOS 14", "Ubuntu 22.04"]),
        "{version}":   f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}",
        "{browser}":   random.choice(["Chrome 124", "Firefox 115", "Safari 17", "Edge 122"]),
        "{duration}":  str(random.randint(10, 120)),
        "{size}":      str(random.randint(10, 100)),
        "{tool}":      random.choice(["Slack", "Salesforce", "HubSpot", "Zapier", "Shopify", "Stripe"]),
        "{product}":   fake.catch_phrase(),
        "{status}":    random.choice(["In Transit", "Out for Delivery", "At Customs", "Processing"]),
        "{address}":   fake.address().replace("\n", ", "),
        "{feature}":   random.choice(["bulk export", "dark mode", "audit logs", "custom reports",
                                       "SSO login", "API webhooks", "mobile notifications"]),
        "{suggestion}":random.choice(["keyboard shortcuts", "batch actions", "CSV preview", "drag-and-drop"]),
        "{plan1}":     random.choice(["Starter", "Basic", "Pro"]),
        "{plan2}":     random.choice(["Business", "Enterprise", "Teams"]),
        "{quarter}":   str(random.randint(1, 4)),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def make_ticket(ticket_id, category_name, config):
    priority   = weighted_priority(config["priority_weights"])
    subject    = fill(random.choice(config["subjects"]))
    body       = fill(random.choice(config["bodies"]))
    resolution = fill(random.choice(config["resolutions"]))

    created_at = fake.date_time_between(start_date="-180d", end_date="now")

    # resolved tickets get a resolution date
    status = random.choices(
        STATUSES,
        weights=[0.10, 0.15, 0.45, 0.20, 0.10],
        k=1
    )[0]
    resolved_at = ""
    if status in ("resolved", "closed"):
        delta_hours = random.randint(1, 72)
        resolved_at = (created_at + timedelta(hours=delta_hours)).strftime("%Y-%m-%d %H:%M:%S")

    word_count = len(body.split())

    return {
        "ticket_id":      f"TK-{ticket_id:06d}",
        "created_at":     created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "resolved_at":    resolved_at,
        "channel":        random.choice(CHANNELS),
        "agent_id":       random.choice(AGENTS),
        "category":       category_name,
        "priority":       priority,
        "status":         status,
        "subject":        subject,
        "body":           body,
        "resolution_note":resolution if status in ("resolved", "closed") else "",
        "word_count":     word_count,
    }


def generate(n_tickets=1000, output_path="data/raw/support_tickets.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # distribute tickets proportionally across categories
    cat_names   = list(CATEGORIES.keys())
    cat_weights = [0.18, 0.22, 0.15, 0.17, 0.15, 0.13]   # billing, technical, account, shipping, product, general
    assignments = random.choices(cat_names, weights=cat_weights, k=n_tickets)

    tickets = []
    for i, cat in enumerate(assignments, start=1):
        tickets.append(make_ticket(i, cat, CATEGORIES[cat]))

    # shuffle so categories are interleaved
    random.shuffle(tickets)

    fieldnames = list(tickets[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)

    print(f"✅  Generated {n_tickets} tickets → {output_path}")

    # ── quick stats ──
    from collections import Counter
    cats  = Counter(t["category"] for t in tickets)
    prios = Counter(t["priority"] for t in tickets)
    stats = Counter(t["status"]   for t in tickets)
    print("\nCategory distribution:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v:>5}  ({v/n_tickets*100:.1f}%)")
    print("\nPriority distribution:")
    for k, v in sorted(prios.items()):
        print(f"  {k:<8} {v:>5}  ({v/n_tickets*100:.1f}%)")
    print("\nStatus distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<15} {v:>5}  ({v/n_tickets*100:.1f}%)")

    return output_path


if __name__ == "__main__":
    generate(n_tickets=1000)