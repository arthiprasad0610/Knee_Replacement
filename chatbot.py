"""
chatbot.py — Rule-based NLP chatbot for knee replacement Q&A.

Architecture:
  - Intent detection via keyword matching (no external API required)
  - 40+ intent categories: pre-op, post-op, pain, exercises, diet, 
    medications, emergency, BMI, doctor recommendations, milestones
  - Returns structured response with intent label and message text
"""

import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini if key is provided
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_enabled = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_enabled = True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

# ---------------------------------------------------------------------------
# Intent Knowledge Base
# Each intent has: keywords (list), response (str), emergency (bool)
# ---------------------------------------------------------------------------

INTENTS = [
    # ── General / Greeting ───────────────────────────────────────────────
    {
        "intent": "greeting",
        "keywords": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "howdy", "greetings"],
        "response": (
            "👋 Hello! I'm KneeBot, your specialized assistant for knee replacement surgery. "
            "I'm here to guide you through your journey—whether you're just considering surgery, "
            "preparing for your date, or in the thick of recovery. <br><br> "
            "Ask me anything about:<br>"
            "• 🏥 <b>Surgical procedures</b> & what to expect<br>"
            "• 📋 <b>Preparation</b> (pre-op) checklists<br>"
            "• 💊 <b>Pain & Swelling</b> management<br>"
            "• 🏋️ <b>Physiotherapy</b> & recovery exercises<br>"
            "• 🚨 <b>Emergency signs</b> to watch for"
        ),
        "emergency": False,
    },
    {
        "intent": "goodbye",
        "keywords": ["bye", "goodbye", "see you", "thanks bye", "take care", "farewell"],
        "response": (
            "👋 Goodbye! Take care of yourself and remember — consistent recovery efforts "
            "make a huge difference. Don't hesitate to return if you have more questions!"
        ),
        "emergency": False,
    },
    {
        "intent": "thanks",
        "keywords": ["thank", "thanks", "thank you", "appreciate", "helpful"],
        "response": (
            "😊 You're very welcome! I'm always here to support your knee replacement journey. "
            "Is there anything else you'd like to know?"
        ),
        "emergency": False,
    },

    # ── What is Knee Replacement ─────────────────────────────────────────
    {
        "intent": "what_is_knee_replacement",
        "keywords": ["what is knee replacement", "knee replacement surgery", "total knee replacement", "tkr", "knee arthroplasty", "what does knee replacement mean"],
        "response": (
            "🦴 <strong>Knee Replacement Surgery (Total Knee Arthroplasty)</strong><br><br>"
            "Knee replacement surgery replaces damaged or worn cartilage and bone surfaces "
            "of the knee joint with artificial components (prosthesis) made of metal alloys "
            "and medical-grade plastic.<br><br>"
            "📌 <strong>Types:</strong><br>"
            "• <b>Total Knee Replacement (TKR)</b> — replaces the entire knee joint<br>"
            "• <b>Partial (Unicompartmental)</b> — only one side of the knee<br>"
            "• <b>Kneecap Replacement</b> — only the undersurface of the kneecap<br><br>"
            "⏱️ Surgery typically takes 1–2 hours. Hospital stay is usually 1–3 days."
        ),
        "emergency": False,
    },

    # ── Candidacy ────────────────────────────────────────────────────────
    {
        "intent": "candidacy",
        "keywords": ["am i candidate", "qualify for knee", "who needs knee replacement", "do i need surgery", "eligibility", "suitable for surgery"],
        "response": (
            "🩺 <strong>Are You a Candidate for Knee Replacement?</strong><br><br>"
            "You may be a good candidate if you have:<br>"
            "✅ Severe knee pain that limits daily activities<br>"
            "✅ Knee pain even at rest or lying down<br>"
            "✅ Knee stiffness that restricts range of motion<br>"
            "✅ Failed conservative treatments (physio, medication, injections)<br>"
            "✅ X-ray evidence of significant joint damage<br><br>"
            "❌ Surgery may not be ideal if you have: active infection, severe obesity (BMI>40), "
            "poor skin condition around the knee, or serious heart/lung issues.<br><br>"
            "👉 Please consult an orthopaedic surgeon for a personalised evaluation."
        ),
        "emergency": False,
    },

    # ── Pre-Surgery Preparation ──────────────────────────────────────────
    {
        "intent": "pre_surgery_prep",
        "keywords": ["before surgery", "pre surgery", "pre-surgery", "prepare for surgery", "pre op", "pre-op", "what to do before", "getting ready for surgery"],
        "response": (
            "📋 <strong>Pre-Surgery Preparation Checklist</strong><br><br>"
            "🏥 <b>Medical:</b><br>"
            "• Complete all pre-op blood tests, ECG, and X-rays<br>"
            "• Disclose all current medications to your surgeon<br>"
            "• Stop blood thinners as directed (usually 5–7 days before)<br>"
            "• Dental work should be done before surgery (infection risk)<br><br>"
            "🏠 <b>Home Setup:</b><br>"
            "• Install grab bars in bathroom and shower<br>"
            "• Remove trip hazards (rugs, cords)<br>"
            "• Arrange a raised toilet seat<br>"
            "• Prepare a recovery area on ground floor if possible<br><br>"
            "💪 <b>Physical:</b><br>"
            "• Strengthen quads and hamstrings pre-op<br>"
            "• Lose weight if BMI > 35 (reduces complications)<br>"
            "• Stop smoking at least 6 weeks before surgery"
        ),
        "emergency": False,
    },

    # ── Post-Surgery Care ────────────────────────────────────────────────
    {
        "intent": "post_surgery_care",
        "keywords": ["after surgery", "post surgery", "post-surgery", "post op", "post-op", "what to do after", "recovery after surgery", "discharge"],
        "response": (
            "🏥 <strong>Post-Surgery Care Guidelines</strong><br><br>"
            "🩹 <b>Wound Care:</b><br>"
            "• Keep incision clean and dry for 48 hours<br>"
            "• Report any redness, warmth, or discharge immediately<br>"
            "• Sutures/staples removed at 10–14 days<br><br>"
            "💊 <b>Medications:</b><br>"
            "• Take prescribed pain medications on schedule<br>"
            "• Blood thinners (anticoagulants) to prevent DVT — very important!<br><br>"
            "🚶 <b>Activity:</b><br>"
            "• Walk with crutches/walker from Day 1<br>"
            "• Elevate leg to reduce swelling (above heart level)<br>"
            "• Ice the knee 20 min every 2 hours<br><br>"
            "📅 <b>Follow-ups:</b><br>"
            "• Week 2: Wound check<br>"
            "• Week 6: Range of motion assessment<br>"
            "• Month 3: Functional assessment<br>"
            "• Year 1: X-ray review"
        ),
        "emergency": False,
    },

    # ── Pain Management ──────────────────────────────────────────────────
    {
        "intent": "pain_management",
        "keywords": ["pain", "pain management", "hurts", "hurt", "painkillers", "medication for pain", "how to manage pain", "knee hurts", "pain relief"],
        "response": (
            "💊 <strong>Pain Management After Knee Replacement</strong><br><br>"
            "📌 <b>Medications (as prescribed):</b><br>"
            "• Paracetamol / Acetaminophen — regular schedule<br>"
            "• NSAIDs (Ibuprofen/Naproxen) — with food<br>"
            "• Opioids — short term only for severe pain<br>"
            "• Nerve blocks / epidural — immediately post-op<br><br>"
            "❄️ <b>Non-Pharmacological:</b><br>"
            "• Ice packs: 15–20 minutes, 3–4 times daily<br>"
            "• Elevation: keep leg higher than heart<br>"
            "• Compression stockings<br>"
            "• TENS therapy (with physio guidance)<br><br>"
            "⚠️ Pain level > 8/10 not controlled by medication needs <b>urgent medical review</b>."
        ),
        "emergency": False,
    },

    # ── Exercises ────────────────────────────────────────────────────────
    {
        "intent": "exercises",
        "keywords": ["exercise", "exercises", "physiotherapy", "physio", "stretches", "rehab", "rehabilitation", "workout", "physical therapy", "flexion", "extension", "range of motion", "squats", "cycling", "swimming", "walk", "walking", "bend", "bending", "stiff", "stiffness", "stretch"],
        "response": (
            "🏋️ <strong>Knee Replacement Rehabilitation Exercises</strong><br><br>"
            "📅 <b>Week 1 (Gentle Activation):</b><br>"
            "• Ankle pumps — 10 reps, 3x/day<br>"
            "• Quad sets (press back of knee to bed) — 10 reps<br>"
            "• Heel slides — 10 reps<br>"
            "• Straight leg raises — 10 reps<br><br>"
            "📅 <b>Week 2–4 (Progressive):</b><br>"
            "• Sitting knee bends (flexion to 90°)<br>"
            "• Short arc quads<br>"
            "• Standing knee bends (with support)<br>"
            "• Step exercises on low step<br><br>"
            "📅 <b>Week 6–12 (Advanced):</b><br>"
            "• Stationary cycling<br>"
            "• Swimming / hydrotherapy<br>"
            "• Mini squats (to 45°)<br>"
            "• Walking on slopes/stairs<br><br>"
            "⚠️ Always perform exercises as guided by your physiotherapist."
        ),
        "emergency": False,
    },

    # ── Diet & Nutrition ─────────────────────────────────────────────────
    {
        "intent": "diet",
        "keywords": ["diet", "food", "nutrition", "eat", "eating", "what to eat", "foods", "weight", "healthy eating", "vitamins", "supplements"],
        "response": (
            "🥗 <strong>Diet & Nutrition for Knee Recovery</strong><br><br>"
            "✅ <b>Eat More Of:</b><br>"
            "• Lean protein (chicken, fish, eggs, lentils) — tissue repair<br>"
            "• Colourful vegetables — anti-inflammatory antioxidants<br>"
            "• Omega-3 rich foods (salmon, walnuts, flaxseed)<br>"
            "• Calcium & Vitamin D (dairy, fortified foods, sunlight)<br>"
            "• Whole grains for sustained energy<br>"
            "• Water — minimum 2L/day for healing<br><br>"
            "❌ <b>Avoid:</b><br>"
            "• Processed foods and trans fats<br>"
            "• Excess sugar (promotes inflammation)<br>"
            "• Alcohol (interacts with medications, slows healing)<br>"
            "• High-sodium foods (increase swelling)<br><br>"
            "💊 <b>Supplements (consult doctor first):</b><br>"
            "• Vitamin C — wound healing<br>"
            "• Iron — if anaemic post-surgery<br>"
            "• Zinc — immune support"
        ),
        "emergency": False,
    },

    # ── Swelling ─────────────────────────────────────────────────────────
    {
        "intent": "swelling",
        "keywords": ["swelling", "swollen", "puffy", "oedema", "edema", "fluid", "inflammation"],
        "response": (
            "🧊 <strong>Managing Swelling After Knee Replacement</strong><br><br>"
            "Swelling is completely normal for 3–6 months after surgery. Here's how to manage it:<br><br>"
            "❄️ <b>RICE Protocol:</b><br>"
            "• <b>Rest</b> — avoid overactivity<br>"
            "• <b>Ice</b> — 15–20 min, wrapped in cloth, 3–4x daily<br>"
            "• <b>Compression</b> — use compression bandage as advised<br>"
            "• <b>Elevation</b> — raise leg above heart level when resting<br><br>"
            "⚠️ <b>See your doctor urgently if swelling is:</b><br>"
            "• Sudden and severe<br>"
            "• Accompanied by calf pain or redness (could be DVT)<br>"
            "• Associated with high fever<br>"
            "• Not improving by Week 12"
        ),
        "emergency": False,
    },

    # ── DVT / Blood Clot ─────────────────────────────────────────────────
    {
        "intent": "dvt",
        "keywords": ["dvt", "blood clot", "deep vein thrombosis", "clot", "pulmonary embolism", "pe", "calf pain", "leg swollen red"],
        "response": (
            "🚨 <strong>DVT (Deep Vein Thrombosis) Warning</strong><br><br>"
            "DVT is one of the most serious post-operative complications. Know the signs:<br><br>"
            "🔴 <b>Symptoms to watch for:</b><br>"
            "• Calf pain, tenderness, or cramps<br>"
            "• One leg significantly more swollen than the other<br>"
            "• Redness or warmth in the calf area<br>"
            "• Shortness of breath (possible pulmonary embolism — EMERGENCY!)<br><br>"
            "✅ <b>Prevention:</b><br>"
            "• Take prescribed anticoagulants (blood thinners) as directed<br>"
            "• Ankle pumping exercises from Day 1<br>"
            "• Walk frequently<br>"
            "• Stay hydrated<br><br>"
            "🚑 <b>If you suspect DVT — seek EMERGENCY medical care immediately!</b>"
        ),
        "emergency": True,
    },

    # ── Infection Signs ───────────────────────────────────────────────────
    {
        "intent": "infection",
        "keywords": ["infection", "infected", "pus", "discharge", "wound", "red", "fever", "hot knee", "warm knee"],
        "response": (
            "🚨 <strong>Signs of Knee Infection — Act Quickly!</strong><br><br>"
            "Prosthetic joint infection (PJI) is a serious complication. Watch for:<br><br>"
            "🔴 <b>Warning Signs:</b><br>"
            "• Fever above 38.5°C (101.3°F)<br>"
            "• Increasing (not decreasing) knee pain after Day 3<br>"
            "• Redness, warmth, and swelling around incision<br>"
            "• Pus or unusual fluid drainage<br>"
            "• Wound edges not closing / reopening<br><br>"
            "⚡ <b>What to do:</b><br>"
            "• Contact your surgical team IMMEDIATELY<br>"
            "• Do NOT wait for your next scheduled appointment<br>"
            "• Go to ER if your surgeon is unreachable<br><br>"
            "🚑 Early treatment (antibiotics/washout surgery) is critical to save the implant."
        ),
        "emergency": True,
    },

    # ── Emergency / Chest Pain ────────────────────────────────────────────
    {
        "intent": "emergency_chest",
        "keywords": ["chest pain", "can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath", "heart attack", "stroke", "collapsed", "unconscious"],
        "response": (
            "🚑 <strong>EMERGENCY — Call 911 / 999 / 112 Immediately!</strong><br><br>"
            "The symptoms you describe are potentially life-threatening.<br><br>"
            "📞 <b>Call emergency services NOW!</b><br>"
            "• Tell them you have had recent knee surgery<br>"
            "• Do not drive yourself<br>"
            "• Stay as calm as possible<br>"
            "• If unconscious — perform CPR if trained<br><br>"
            "⚡ Do NOT delay seeking emergency care!"
        ),
        "emergency": True,
    },

    # ── BMI & Weight ──────────────────────────────────────────────────────
    {
        "intent": "bmi",
        "keywords": ["bmi", "body mass index", "overweight", "obese", "obesity", "weight loss", "ideal weight"],
        "response": (
            "⚖️ <strong>BMI & Knee Replacement</strong><br><br>"
            "📊 <b>BMI Categories:</b><br>"
            "• Under 18.5 — Underweight<br>"
            "• 18.5–24.9 — Healthy Weight ✅<br>"
            "• 25–29.9 — Overweight ⚠️<br>"
            "• 30–34.9 — Obese (Class I)<br>"
            "• 35–39.9 — Obese (Class II) — higher surgical risk<br>"
            "• 40+ — Morbidly Obese — significantly higher risk<br><br>"
            "💡 <b>Why it matters:</b><br>"
            "• Higher BMI → increased risk of infection, blood clots, implant loosening<br>"
            "• Even 5–10% weight loss before surgery significantly reduces complications<br>"
            "• Many surgeons recommend BMI < 35–40 before operating<br><br>"
            "🥗 Work with a dietitian for a structured weight loss plan before your surgery."
        ),
        "emergency": False,
    },

    # ── Recovery Timeline ────────────────────────────────────────────────
    {
        "intent": "recovery_timeline",
        "keywords": ["recovery time", "how long", "when will i recover", "timeline", "recovery period", "full recovery", "weeks", "months"],
        "response": (
            "📅 <strong>Knee Replacement Recovery Timeline</strong><br><br>"
            "🏥 <b>Days 1–3</b> — Hospital stay, pain management, start walking with support<br>"
            "🏠 <b>Week 1–2</b> — Home recovery, wound care, basic exercises, crutches<br>"
            "🚶 <b>Week 3–6</b> — Increased mobility, reduce crutch use, flexion to 90°<br>"
            "🏋️ <b>Month 2–3</b> — More intensive physio, walking without aids, driving may resume<br>"
            "🚴 <b>Month 3–6</b> — Swimming/cycling, stairs independently, most daily activities<br>"
            "🎯 <b>Month 6–12</b> — Near-full function for most patients<br>"
            "🏆 <b>Year 1–2</b> — Maximum improvement, implant fully integrated<br><br>"
            "📌 Full recovery varies by age, fitness level, and adherence to physio. "
            "Most patients achieve 90–95% of normal function."
        ),
        "emergency": False,
    },

    # ── Walking & Mobility ───────────────────────────────────────────────
    {
        "intent": "walking",
        "keywords": ["walk", "walking", "mobility", "can i walk", "when can i walk", "crutches", "walker", "stairs"],
        "response": (
            "🚶 <strong>Walking After Knee Replacement</strong><br><br>"
            "📌 Most patients begin walking <b>within 24 hours</b> of surgery!<br><br>"
            "🗓️ <b>Walking Progression:</b><br>"
            "• Day 1: Short distances with walking frame/crutches<br>"
            "• Week 1–2: Increasing distances, 10–15 min sessions<br>"
            "• Week 3–4: May reduce to one crutch<br>"
            "• Week 6–8: Walk without aids (most patients)<br>"
            "• Month 3: 30+ min walks, gentle hiking<br><br>"
            "⚠️ <b>Tips:</b><br>"
            "• Always wear supportive, non-slip footwear<br>"
            "• Walk on flat surfaces initially<br>"
            "• When climbing stairs: 'Good leg goes up first, bad leg comes down first'<br>"
            "• Stop if you feel sharp pain or instability"
        ),
        "emergency": False,
    },

    # ── Driving ──────────────────────────────────────────────────────────
    {
        "intent": "driving",
        "keywords": ["drive", "driving", "car", "when can i drive"],
        "response": (
            "🚗 <strong>Driving After Knee Replacement</strong><br><br>"
            "📌 <b>General guidance:</b><br>"
            "• Right knee surgery: typically 6–8 weeks before driving<br>"
            "• Left knee surgery (automatic car): often 4–6 weeks<br><br>"
            "✅ <b>Before driving you need:</b><br>"
            "• Your surgeon's clearance<br>"
            "• Ability to do an emergency stop quickly<br>"
            "• Not taking strong opioid pain medications<br>"
            "• Adequate range of motion and strength<br><br>"
            "⚠️ Check your car insurance policy — some have post-surgery clauses."
        ),
        "emergency": False,
    },

    # ── Sleep ────────────────────────────────────────────────────────────
    {
        "intent": "sleep",
        "keywords": ["sleep", "sleeping", "lying down", "position to sleep", "can't sleep", "insomnia"],
        "response": (
            "😴 <strong>Sleeping After Knee Replacement</strong><br><br>"
            "🛏️ <b>Best positions:</b><br>"
            "• On your back with a pillow under the operated leg<br>"
            "• Gradually try sleeping on your side (weeks 4–6)<br>"
            "• Avoid pillow under the KNEE — causes contractions<br><br>"
            "💡 <b>Tips for better sleep:</b><br>"
            "• Take pain medication 30–60 min before bed<br>"
            "• Ice the knee before sleeping<br>"
            "• Keep bedroom cool<br>"
            "• Avoid caffeine after 2pm<br><br>"
            "📌 Sleep disturbance is very common in weeks 1–3 and usually improves."
        ),
        "emergency": False,
    },

    # ── Work Return ──────────────────────────────────────────────────────
    {
        "intent": "return_to_work",
        "keywords": ["return to work", "go back to work", "work", "office", "job", "when can i work"],
        "response": (
            "💼 <strong>Returning to Work After Knee Replacement</strong><br><br>"
            "📌 Timelines depend on job type:<br><br>"
            "🖥️ <b>Desk/office job:</b> 4–8 weeks<br>"
            "🚗 <b>Driving-heavy job:</b> 6–10 weeks<br>"
            "🧍 <b>Standing/walking job:</b> 10–16 weeks<br>"
            "🏗️ <b>Heavy manual labour:</b> 16–26 weeks (may need role change)<br><br>"
            "✅ Discuss with your surgeon and consider phased return to work programs."
        ),
        "emergency": False,
    },

    # ── Sports & Recreation ──────────────────────────────────────────────
    {
        "intent": "sports",
        "keywords": ["sport", "sports", "swim", "swimming", "cycling", "golf", "tennis", "running", "yoga", "pilates", "fitness", "gym"],
        "response": (
            "🏊 <strong>Sports & Recreation After Knee Replacement</strong><br><br>"
            "✅ <b>Generally Recommended:</b><br>"
            "• Swimming (6–12 weeks depending on wound healing)<br>"
            "• Cycling (stationary: 6 weeks; road: 3 months)<br>"
            "• Golf (3–6 months)<br>"
            "• Walking, hiking (gentle terrain: 3 months)<br>"
            "• Yoga / Pilates (modified, 3+ months)<br><br>"
            "⚠️ <b>Proceed with Caution:</b><br>"
            "• Tennis / Doubles (6 months+)<br>"
            "• Skiing (12+ months, with surgeon approval)<br><br>"
            "❌ <b>Generally Avoid:</b><br>"
            "• High-impact running / jogging<br>"
            "• Contact sports<br>"
            "• Squash, basketball, football"
        ),
        "emergency": False,
    },

    # ── Implant Longevity ────────────────────────────────────────────────
    {
        "intent": "implant",
        "keywords": ["implant", "prosthesis", "how long does implant last", "revision surgery", "implant longevity", "artificial knee"],
        "response": (
            "🔩 <strong>Knee Implant Longevity</strong><br><br>"
            "📊 Modern knee implants last:<br>"
            "• <b>15+ years</b> in 90–95% of patients<br>"
            "• <b>20+ years</b> in ~80% of patients<br><br>"
            "📌 <b>Factors affecting longevity:</b><br>"
            "• Body weight (lower BMI → less wear)<br>"
            "• Activity level (high-impact sports increase wear)<br>"
            "• Bone quality / osteoporosis<br>"
            "• Surgical technique<br>"
            "• Infection (major cause of early failure)<br><br>"
            "🔄 Revision surgery is available when implants wear out, though more complex."
        ),
        "emergency": False,
    },

    # ── Complications ────────────────────────────────────────────────────
    {
        "intent": "complications",
        "keywords": ["complications", "risks", "side effects", "what can go wrong", "problems", "failure"],
        "response": (
            "⚠️ <strong>Possible Complications of Knee Replacement</strong><br><br>"
            "🔴 <b>Serious (rare but important):</b><br>"
            "• Deep vein thrombosis (DVT) / pulmonary embolism<br>"
            "• Prosthetic joint infection<br>"
            "• Implant loosening or fracture<br>"
            "• Nerve or blood vessel injury<br><br>"
            "🟡 <b>Moderate (manageable):</b><br>"
            "• Prolonged swelling and stiffness<br>"
            "• Scar tissue formation (arthrofibrosis)<br>"
            "• Anaesthetic reactions<br>"
            "• Wound healing problems<br><br>"
            "🟢 <b>Common (expected):</b><br>"
            "• Pain and bruising (first few weeks)<br>"
            "• Numbness around scar<br>"
            "• Clicking/clunking sounds from implant<br><br>"
            "📌 Overall complication rate is low (~2–5%) in experienced centres."
        ),
        "emergency": False,
    },

    # ── Milestones ───────────────────────────────────────────────────────
    {
        "intent": "milestones",
        "keywords": ["milestone", "milestones", "day 7", "day 14", "week 6", "3 month", "6 month", "progress", "goals"],
        "response": (
            "🏆 <strong>Recovery Milestones</strong><br><br>"
            "🎯 <b>Day 7</b> — Walk 100m, knee flexion ~60°, wound check<br>"
            "🎯 <b>Day 14</b> — Suture removal, walk 200m, most require only one crutch<br>"
            "🎯 <b>Week 6</b> — Knee flexion 90°+, may drive, reduced reliance on aids<br>"
            "🎯 <b>Month 3</b> — Return to most daily activities, gentle sports<br>"
            "🎯 <b>Month 6</b> — Most feel near-normal, swelling largely resolved<br>"
            "🎯 <b>Month 12</b> — Annual X-ray review, maximum function achieved<br><br>"
            "📌 Track your progress in the <a href='/dashboard' style='color:#4FC3F7'>Recovery Dashboard</a>!"
        ),
        "emergency": False,
    },

    # ── Doctor Consultation ──────────────────────────────────────────────
    {
        "intent": "when_see_doctor",
        "keywords": ["see doctor", "call doctor", "when to call", "urgent", "worried", "concerned", "should i be worried", "is this normal"],
        "response": (
            "🩺 <strong>When to Contact Your Doctor</strong><br><br>"
            "📞 <b>Call your surgical team within 24 hours if:</b><br>"
            "• Wound is oozing or looks infected<br>"
            "• Pain is not controlled by medications<br>"
            "• Significant swelling in one leg (DVT risk)<br>"
            "• Numbness or weakness beyond expected<br><br>"
            "🚑 <b>Go to Emergency immediately if:</b><br>"
            "• Chest pain or difficulty breathing<br>"
            "• High fever (38.5°C+) with knee redness<br>"
            "• Suspected blood clot<br>"
            "• Inability to bear any weight<br>"
            "• Wound completely opens up<br><br>"
            "✅ When in doubt — always call your surgeon's team. Better safe than sorry!"
        ),
        "emergency": False,
    },

    # ── Physiotherapy ────────────────────────────────────────────────────
    {
        "intent": "physiotherapy",
        "keywords": ["physiotherapist", "physio", "physical therapy", "pt", "therapist", "appointment", "session"],
        "response": (
            "🏃 <strong>Physiotherapy After Knee Replacement</strong><br><br>"
            "📌 <b>Typically starts:</b> 24–48 hours after surgery (in hospital)<br><br>"
            "📅 <b>Typical Schedule:</b><br>"
            "• Week 1–2: Daily (in hospital then home visits)<br>"
            "• Week 3–6: 2–3x per week (outpatient clinic)<br>"
            "• Week 6–12: 1–2x per week<br>"
            "• Month 3–6: Weekly or as needed<br><br>"
            "🎯 <b>Goals of Physiotherapy:</b><br>"
            "• Regain knee flexion (target 120°+)<br>"
            "• Strengthen quadriceps and hamstrings<br>"
            "• Improve balance and proprioception<br>"
            "• Reduce swelling<br>"
            "• Return to normal gait<br><br>"
            "💡 Consistency with home exercises between sessions is crucial!"
        ),
        "emergency": False,
    },

    # ── AI Prediction ────────────────────────────────────────────────────
    {
        "intent": "prediction",
        "keywords": ["predict", "prediction", "ai", "artificial intelligence", "machine learning", "forecast", "expected recovery"],
        "response": (
            "🤖 <strong>AI Recovery Prediction</strong><br><br>"
            "Our platform uses a statistical model to estimate your recovery trajectory based on:<br><br>"
            "📊 <b>Input Factors:</b><br>"
            "• Age and pre-operative BMI<br>"
            "• Pain levels and mobility scores<br>"
            "• Daily exercise completion rates<br>"
            "• Swelling trends<br>"
            "• Medication adherence<br><br>"
            "📈 <b>How it works:</b><br>"
            "The model compares your recovery data against typical recovery curves. "
            "Patients in the top third of outcomes ('fast recovery') typically show "
            "steady pain reduction and increasing walking distance in the first 4 weeks.<br><br>"
            "👉 Log your daily recovery data in the <a href='/dashboard' style='color:#4FC3F7'>Dashboard</a> for personalised predictions!"
        ),
        "emergency": False,
    },

    # ── Showering & Bathing ───────────────────────────────────────────────
    {
        "intent": "showering",
        "keywords": ["shower", "showering", "bath", "bathing", "wash", "washing", "wet", "water on wound", "get the wound wet", "get wound wet", "wound wet"],
        "response": (
            "🚿 <strong>Showering & Bathing After Surgery</strong><br><br>"
            "🩹 <b>General Timeline:</b><br>"
            "• <b>First 48 hours:</b> Keep the incision completely dry. Do not shower.<br>"
            "• <b>After 48 hours (with waterproof dressing):</b> You can typically take a quick shower if your surgeon used a waterproof dressing. Do not remove the dressing.<br>"
            "• <b>Without waterproof dressing / Staples exposed:</b> Avoid direct water until staples/sutures are removed (usually 10–14 days). Use sponge baths.<br><br>"
            "⚠️ <b>Important Precautions:</b><br>"
            "• <b>No soaking:</b> Do NOT submerge your knee in a bath, hot tub, or swimming pool until the wound is fully healed (typically 4–6 weeks).<br>"
            "• <b>Safety first:</b> Use a shower chair or grab bars to prevent slipping. Do not bend excessively to wash your leg.<br>"
            "• <b>Dry gently:</b> Pat the area dry with a clean towel; do not rub the incision."
        ),
        "emergency": False,
    },

    # ── Ice vs Heat ───────────────────────────────────────────────────────
    {
        "intent": "ice_vs_heat",
        "keywords": ["ice or heat", "heat or ice", "heating pad", "hot pack", "warm compress", "use heat", "when to use heat"],
        "response": (
            "❄️🔥 <strong>Ice vs. Heat for Knee Recovery</strong><br><br>"
            "❄️ <b>Ice (Recommended for early recovery):</b><br>"
            "• <b>When:</b> First 3–6 months post-surgery, especially after exercises or when swollen.<br>"
            "• <b>Why:</b> Constricts blood vessels, reduces swelling, and numbs pain.<br>"
            "• <b>How:</b> 15–20 minutes at a time, wrapping the ice pack in a thin towel (never apply ice directly to skin).<br><br>"
            "🔥 <b>Heat (Use with caution):</b><br>"
            "• <b>When:</b> Only after the surgical swelling has completely resolved (usually 3–4 months post-op) and if you have chronic muscle stiffness.<br>"
            "• <b>Why:</b> Relaxes muscles and increases blood circulation.<br>"
            "• <b>Caution:</b> Applying heat too early will **increase swelling and inflammation** in the surgical area.<br><br>"
            "👉 Rule of thumb: If in doubt, stick to <b>Ice</b>!"
        ),
        "emergency": False,
    },

    # ── Kneeling ──────────────────────────────────────────────────────────
    {
        "intent": "kneeling",
        "keywords": ["kneel", "kneeling", "on my knees", "get on knees", "kneel down"],
        "response": (
            "🧎 <strong>Kneeling After Knee Replacement</strong><br><br>"
            "📌 <b>Can you kneel?</b><br>"
            "Yes, technically it is safe to kneel on your artificial knee once the wound is fully healed (typically after 3 months). "
            "It will not damage the prosthesis.<br><br>"
            "⚠️ <b>What to expect:</b><br>"
            "• <b>Discomfort:</b> About 60–80% of patients find kneeling uncomfortable or feel a strange sensation due to scar tissue and nerve sensitivity.<br>"
            "• <b>Numbness:</b> Mild numbness on the outer side of the scar is normal and can make kneeling feel unusual.<br><br>"
            "💡 <b>Tips for kneeling:</b><br>"
            "• Use a soft cushion, kneeling pad, or folded towel.<br>"
            "• Practice kneeling on a soft surface (like a mattress or carpet) first.<br>"
            "• Try putting weight on the surrounding areas rather than directly on the kneecap."
        ),
        "emergency": False,
    },

    # ── Travel & Airport Security ─────────────────────────────────────────
    {
        "intent": "travel_airport",
        "keywords": ["fly", "flying", "airplane", "airport", "metal detector", "metal detectors", "travel", "travelling", "vacation"],
        "response": (
            "✈️ <strong>Travel & Airport Security After Surgery</strong><br><br>"
            "🔒 <b>Airport Metal Detectors:</b><br>"
            "• Yes, your metal joint implant will likely set off airport security scanners.<br>"
            "• Inform the security officers before walking through. They are very used to this and will scan you with a hand wand or body scanner.<br>"
            "• You do **not** need a doctor's note or certificate; security officers do not rely on them anyway.<br><br>"
            "🚗✈️ <b>Long Trips & DVT Risk:</b><br>"
            "• <b>Short flights/trips:</b> Safe after 4–6 weeks.<br>"
            "• <b>Long-haul flights (>4 hours):</b> Avoid for at least 6–12 weeks to prevent deep vein thrombosis (blood clots).<br><br>"
            "💡 <b>Travel tips:</b><br>"
            "• Walk around every 45–60 minutes during long flights or car trips.<br>"
            "• Wear compression stockings during travel.<br>"
            "• Stay well-hydrated."
        ),
        "emergency": False,
    },

    # ── Scarring & Numbness ───────────────────────────────────────────────
    {
        "intent": "scarring_numbness",
        "keywords": ["scar", "scars", "numb", "numbness", "skin numb", "incision scar", "scar tissue"],
        "response": (
            "🩹 <strong>Scarring & Numbness Around the Incision</strong><br><br>"
            "📌 <b>Numbness:</b><br>"
            "• It is extremely common to feel numbness on the **outer (lateral) side** of your knee incision. "
            "This happens because small skin nerves are cut during the surgical entry.<br>"
            "• The numb area will gradually shrink over 1–2 years, but some mild numbness may be permanent. This is harmless and does not affect joint function.<br><br>"
            "🧴 <b>Scar Care (Once fully healed / scab-free):</b><br>"
            "• <b>Massage:</b> Gently massage the scar in circular motions using moisturiser or vitamin E oil to break up scar tissue adhesions (start around Week 4–6).<br>"
            "• <b>Sun protection:</b> Keep the scar covered or apply sunscreen (SPF 30+) for the first year to prevent dark discolouration."
        ),
        "emergency": False,
    },

    # ── Clicking & Clunking ───────────────────────────────────────────────
    {
        "intent": "clicking_sound",
        "keywords": ["clicking", "clunking", "popping", "noise", "sound in knee", "knee clicking", "implant noise", "cracking"],
        "response": (
            "🔊 <strong>Clicking or Clunking Sound in the Knee</strong><br><br>"
            "📌 <b>Is it normal?</b><br>"
            "Yes! Hearing or feeling a clicking, clunking, or popping sound from your new knee is **completely normal** and occurs in up to 80% of patients.<br><br>"
            "💡 <b>Why it happens:</b><br>"
            "• The artificial metal and plastic components are contacting each other during movement.<br>"
            "• As your quadriceps muscles strengthen and swelling decreases, the sound usually becomes quieter and less noticeable.<br><br>"
            "⚠️ <b>When to worry:</b><br>"
            "Contact your surgeon only if the clicking is accompanied by:<br>"
            "• Severe pain<br>"
            "• A feeling of the joint locking or giving way<br>"
            "• Sudden swelling or instability"
        ),
        "emergency": False,
    },

    # ── Default Fallback ──────────────────────────────────────────────────
    {
        "intent": "unknown",
        "keywords": [],  # Catches anything not matched above
        "response": (
            "🤔 I'm not exactly sure about that, but let's see if I can help you find what you need. "
            "My knowledge is focused on <b>knee replacement surgery</b> and <b>rehabilitation</b>.<br><br>"
            "Try asking about:<br>"
            "• <b>Preparation:</b> \"How do I prepare for surgery?\"<br>"
            "• <b>Pain:</b> \"How do I manage pain?\"<br>"
            "• <b>Exercises:</b> \"What exercises can I do at home?\"<br>"
            "• <b>Emergency:</b> \"What are the signs of infection?\"<br><br>"
            "Or use the suggested topics in the sidebar!"
        ),
        "emergency": False,
    },
]


# ---------------------------------------------------------------------------
# Chatbot Engine
# ---------------------------------------------------------------------------

def detect_intent(user_message: str) -> dict:
    """
    Detect the intent of a user message using keyword matching.
    Returns the matched intent dict.
    """
    normalized = user_message.lower().strip()

    best_match = None
    best_score = 0

    for intent_obj in INTENTS:
        if not intent_obj["keywords"]:
            continue  # Skip the fallback intent during matching
        for keyword in intent_obj["keywords"]:
            # Use regex for word boundaries to avoid partial matches (e.g. "hi" in "hiking")
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, normalized):
                # Longer keyword match = better score
                score = len(keyword)
                if score > best_score:
                    best_score = score
                    best_match = intent_obj

    return best_match if best_match else next(i for i in INTENTS if i["intent"] == "unknown")


SYSTEM_INSTRUCTION = (
    "You are KneeBot, a specialized, warm, and highly professional AI assistant for knee replacement surgery (total and partial arthroplasty).\n"
    "Your goal is to guide patients through all stages of their knee replacement journey, including pre-surgery preparation, "
    "post-operative care, exercises (rehabilitation/physiotherapy), pain and swelling management, nutrition/diet, sleep, recovery milestones, and return to work/activities.\n\n"
    "CRITICAL GUIDELINES:\n"
    "1. Medical Tone: Provide clear, medically accurate, reassuring, and practical information. Always keep a professional health-advisory stance.\n"
    "2. Formatting: You MUST format your responses strictly in clean HTML tags for direct rendering in an innerHTML chat bubble. "
    "Do NOT use markdown. Do NOT use markdown code block formatting (like ```html ... ```). "
    "Use <p> for paragraphs, <br> for line breaks, <b> or <strong> for bolding, and <ul> with <li> for lists. Keep HTML well-formed.\n"
    "3. Medical Disclaimer: Where relevant (especially for medications, diagnostics, or when the user asks if something is normal), "
    "remind the user to consult their orthopaedic surgeon, doctor, or physiotherapist for personalized clinical advice.\n"
    "4. Emergency Identification: If the user describes emergency symptoms (e.g. chest pain, extreme shortness of breath, sudden severe swelling/redness in the calf, high fever >38.5C, or the wound opening), "
    "immediately respond with clear instructions to seek emergency services (like 911) or contact their surgeon immediately. Use a red alert visual style or clear bold formatting for emergency callouts.\n"
    "5. Contextual Follow-up: You have access to the conversation history. Keep prior answers in mind and answer follow-up queries naturally based on context."
)


def check_sub_intents(user_message: str, last_bot_intent: str) -> str:
    """
    Check if the user is asking a specific follow-up question related to the previous topic.
    Returns the custom sub-response HTML string if a match is found, otherwise None.
    """
    normalized = user_message.lower().strip()

    # -- Sub-intents for DIET --
    is_diet_context = (last_bot_intent == "diet") or any(kw in normalized for kw in ["diet", "food", "nutrition", "eat", "eating", "foods"])
    if is_diet_context:
        if any(w in normalized for w in ["vegetable", "vegetables", "veggies", "green", "greens", "spinach", "broccoli", "carrot", "carrots", "cabbage", "cauliflower", "peas", "plant"]):
            return (
                "🥗 <strong>Recommended Vegetables for Knee Recovery</strong><br><br>"
                "Including anti-inflammatory and vitamin-rich vegetables in your diet is vital for healing:<br><br>"
                "• 🥦 <b>Broccoli & Brussels Sprouts:</b> Rich in sulforaphane, which helps block enzymes that cause joint pain and inflammation.<br>"
                "• 🥬 <b>Spinach, Kale & Swiss Chard:</b> Packed with Vitamin C and antioxidants that support collagen production for tendon and cartilage repair.<br>"
                "• 🫑 <b>Bell Peppers (Red/Yellow):</b> High in Vitamin C, crucial for immune health and wound healing.<br>"
                "• 🥕 <b>Carrots & Sweet Potatoes:</b> Excellent sources of Vitamin A (Beta-carotene), which assists in cell growth and tissue repair.<br>"
                "• 🧄 <b>Garlic & Onions:</b> Contain diallyl disulfide, an anti-inflammatory compound that limits damage from inflammatory cytokines.<br><br>"
                "👉 <i>Tip: Aim for a colorful plate! Try to steam or lightly cook vegetables to preserve their nutrients.</i>"
            )
        elif any(w in normalized for w in ["protein", "proteins", "meat", "chicken", "fish", "eggs", "lentils", "tofu", "cottage cheese", "egg", "beef", "pork"]):
            return (
                "🥩 <strong>Protein Sources for Muscle & Wound Healing</strong><br><br>"
                "Protein is the building block of repair. Aim for lean sources to build muscle and repair tissue:<br><br>"
                "• 🍗 <b>Lean Poultry (Chicken & Turkey):</b> Essential amino acids for tissue reconstruction.<br>"
                "• 🐟 <b>Fatty Fish (Salmon, Mackerel, Sardines):</b> Provides clean protein plus high levels of anti-inflammatory Omega-3 fatty acids.<br>"
                "• 🥚 <b>Eggs:</b> Highly bioavailable protein containing zinc and vitamin D.<br>"
                "• 🧆 <b>Legumes & Lentils:</b> Plant-based proteins rich in zinc and fiber (helps prevent constipation from pain meds).<br>"
                "• 🥛 <b>Dairy (Greek Yogurt & Cottage Cheese):</b> High protein plus calcium and phosphorus for bone strength."
            )
        elif any(w in normalized for w in ["avoid", "bad", "no eat", "limit", "sugar", "alcohol", "processed", "fat", "trans fat", "sodium", "salty"]):
            return (
                "🚫 <strong>Foods to Limit or Avoid During Recovery</strong><br><br>"
                "Certain foods can slow down your recovery by promoting inflammation or swelling:<br><br>"
                "• 🍭 <b>Sugary Foods & Drinks:</b> High sugar triggers the release of inflammatory cytokines.<br>"
                "• 🍟 <b>Processed Foods & Trans Fats:</b> Chips, fried foods, and packaged snacks increase systemic inflammation.<br>"
                "• 🍺 <b>Alcohol:</b> Dehydrates your body, slows down protein synthesis, and has dangerous interactions with pain medications.<br>"
                "• 🥨 <b>High-Sodium (Salty) Foods:</b> Promotes fluid retention and increases post-operative leg and knee swelling."
            )

    # -- Sub-intents for EXERCISES --
    is_exercise_context = (last_bot_intent == "exercises") or any(kw in normalized for kw in ["exercise", "exercises", "physiotherapy", "physio", "stretches", "rehab", "workout"])
    if is_exercise_context:
        if any(w in normalized for w in ["week 1", "first week", "ankle pumps", "quad sets", "gentle", "day 1", "day 2", "day 3", "how to start"]):
            return (
                "🏋️ <strong>Week 1 Recovery Exercises (Gentle Activation)</strong><br><br>"
                "Do these exercises 2–3 times a day on a flat bed:<br><br>"
                "• <b>Ankle Pumps:</b> Move your ankle up and down (like pressing a pedal). Do 10–20 reps to promote blood circulation and prevent DVT.<br>"
                "• <b>Quad Sets:</b> Tighten your thigh muscle (quadricep) by pressing the back of your knee flat against the bed. Hold for 5 seconds. Repeat 10 times.<br>"
                "• <b>Straight Leg Raises:</b> Tighten your thigh and lift your leg 6 inches off the bed. Hold for 3 seconds, then lower. Repeat 10 times.<br>"
                "• <b>Heel Slides:</b> Gently slide your heel toward your buttocks, bending your knee as far as comfortable. Slide back. Repeat 10 times."
            )
        elif any(w in normalized for w in ["week 2", "week 3", "week 4", "progress", "sitting", "sitting bends", "chair"]):
            return (
                "🚶 <strong>Week 2–4 Recovery Exercises (Progressive Strength)</strong><br><br>"
                "As swelling starts to settle, progress your mobility:<br><br>"
                "• <b>Sitting Knee Bends:</b> Sit on a high chair and slide your foot backward under the chair to increase your flexion angle (aim for 90°).<br>"
                "• <b>Short Arc Quads:</b> Place a rolled towel under your knee. Straighten your knee and lift your foot. Repeat 10 times.<br>"
                "• <b>Standing Knee Bends:</b> Hold onto a stable counter or walker, and lift your foot behind you, bending your knee to a 90° angle. Repeat 10 times."
            )
        elif any(w in normalized for w in ["week 6", "advanced", "cycle", "cycling", "swim", "swimming", "squats", "stairs"]):
            return (
                "🚴 <strong>Week 6–12 Advanced Exercises</strong><br><br>"
                "Move to active conditioning once cleared by your physio:<br><br>"
                "• <b>Stationary Cycling:</b> Start with low resistance. Even partial rotations back and forth help range of motion.<br>"
                "• <b>Mini Squats:</b> Hold a stable rail and lower your hips slightly (no more than 45°). Keep your knees aligned over your toes.<br>"
                "• <b>Hydrotherapy/Swimming:</b> Excellent low-impact exercise once your surgical incision is completely healed and dry."
            )

    # -- Sub-intents for PAIN MANAGEMENT --
    is_pain_context = (last_bot_intent == "pain_management") or any(kw in normalized for kw in ["pain", "hurts", "hurt", "painkillers", "pain relief"])
    if is_pain_context:
        if any(w in normalized for w in ["medication", "medications", "meds", "med", "painkillers", "opioids", "pills", "paracetamol", "ibuprofen", "drugs"]):
            return (
                "💊 <strong>Pain Medications After Knee Replacement</strong><br><br>"
                "A combination of medications is typically prescribed to manage pain effectively:<br><br>"
                "• <b>Paracetamol/Acetaminophen:</b> Taken regularly to provide a baseline level of pain relief.<br>"
                "• <b>NSAIDs (e.g., Ibuprofen, Celecoxib):</b> Helps reduce inflammation and swelling alongside pain relief. Always take with food.<br>"
                "• <b>Opioids (e.g., Tramadol, Oxycodone):</b> Strong medications reserved for severe pain (typically the first 3–7 days). Use with caution due to addiction and constipation risks.<br><br>"
                "⚠️ <i>Never exceed the prescribed dosage and consult your doctor before mixing over-the-counter pain relievers.</i>"
            )
        elif any(w in normalized for w in ["ice", "icing", "elevate", "elevation", "stockings", "compression", "non-drug", "cold packs"]):
            return (
                "❄️ <strong>Non-Drug Pain Management Techniques</strong><br><br>"
                "You can significantly lower your pain levels using physical therapy methods:<br><br>"
                "• <b>Elevation:</b> Elevate your leg so that your knee and ankle are *higher than your heart*. This uses gravity to drain fluid and reduce throbbing pain.<br>"
                "• <b>Icing (Cryotherapy):</b> Apply an ice pack wrapped in a damp towel for 15–20 minutes every 2 hours. This numbs the nerves and constricts inflamed tissues.<br>"
                "• <b>Compression Stockings:</b> Wear your TED hose as directed to prevent fluid pooling in your lower leg."
            )

    return None


def get_response(user_message: str, history=None) -> dict:
    """
    Main chatbot function. Accepts text input and history, returns response dict.
    Uses Gemini Generative AI if enabled, otherwise falls back to local rules-based engine.

    Returns:
        {
          "intent": str,
          "response": str,
          "emergency": bool,
          "timestamp": str
        }
    """
    global gemini_enabled, GEMINI_API_KEY
    
    # Check if a key was set dynamically in environment since last call
    if not gemini_enabled:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            try:
                genai.configure(api_key=env_key)
                GEMINI_API_KEY = env_key
                gemini_enabled = True
                print("Gemini API configured dynamically via get_response.")
            except Exception as e:
                print(f"Error configuring Gemini API dynamically: {e}")

    # Quick symptom scan for user safety
    is_emergency = check_emergency_symptoms(user_message)

    # Determine the last bot intent from history
    last_bot_intent = None
    if history:
        for msg in reversed(history):
            role = getattr(msg, "role", None)
            if role is None and isinstance(msg, dict):
                role = msg.get("role")
            
            if role == "bot":
                last_bot_intent = getattr(msg, "intent", None)
                if last_bot_intent is None and isinstance(msg, dict):
                    last_bot_intent = msg.get("intent")
                break

    if gemini_enabled:
        try:
            # Build conversation history in Gemini format
            contents = []
            if history:
                for msg in history:
                    role = "user" if msg.role == "user" else "model"
                    contents.append({
                        "role": role,
                        "parts": [{"text": msg.content}]
                    })
            
            # Add the current user query
            contents.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })

            # Call Gemini
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_INSTRUCTION
            )
            response = model.generate_content(contents)
            response_text = response.text.strip()
            
            # Perform additional check on generated text for emergency indicators
            if not is_emergency:
                is_emergency = check_emergency_symptoms(response_text)

            # Let's clean up any markdown blocks if the LLM output it by accident
            # e.g., ```html ... ``` or ``` ... ```
            if response_text.startswith("```html"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Return the response structure
            return {
                "intent": "gemini",
                "response": response_text,
                "emergency": is_emergency,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Catch Gemini errors gracefully and continue with local engine
            print(f"Gemini generation error: {e}. Falling back to keyword engine.")
            sub_resp = check_sub_intents(user_message, last_bot_intent)
            if sub_resp:
                intent_name = last_bot_intent if last_bot_intent else "diet"
                response_content = sub_resp
                emergency_flag = is_emergency
            else:
                fallback_res = detect_intent(user_message)
                intent_name = fallback_res["intent"]
                response_content = fallback_res["response"]
                emergency_flag = is_emergency or fallback_res.get("emergency", False)
                
            error_notice = (
                "<div class='gemini-info-notice' style='margin-bottom: 12px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-left: 3px solid #E53935; border-radius: 4px; font-size: 0.9em;'>"
                "⚠️ <b>KneeBot Status:</b> Offline fallback activated due to API connection error. Answers may be limited."
                "</div>"
            )
            return {
                "intent": intent_name,
                "response": error_notice + response_content,
                "emergency": emergency_flag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Keyword engine fallback
    sub_resp = check_sub_intents(user_message, last_bot_intent)
    if sub_resp:
        intent_name = last_bot_intent if last_bot_intent else "diet"
        response_content = sub_resp
        emergency_flag = is_emergency
    else:
        intent_obj = detect_intent(user_message)
        intent_name = intent_obj["intent"]
        response_content = intent_obj["response"]
        emergency_flag = is_emergency or intent_obj.get("emergency", False)

    notice = (
        "<div class='gemini-info-notice' style='margin-bottom: 12px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-left: 3px solid #FFB300; border-radius: 4px; font-size: 0.9em;'>"
        "💡 <b>KneeBot Tip:</b> To enable advanced conversational AI and follow-up questions with Gemini, configure your <code>GEMINI_API_KEY</code> in the <code>.env</code> file."
        "</div>"
    )
    return {
        "intent": intent_name,
        "response": notice + response_content,
        "emergency": emergency_flag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



def check_emergency_symptoms(text: str) -> bool:
    """
    Quick check for emergency-flagged content in any text (e.g., recovery notes).
    Returns True if emergency symptoms are detected.
    """
    emergency_keywords = [
        "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
        "shortness of breath", "heart attack", "stroke", "collapsed", "unconscious",
        "pus", "high fever", "blood clot", "dvt", "severe pain",
        "wound opened", "wound reopened", "unable to walk",
    ]
    normalized = text.lower()
    return any(kw in normalized for kw in emergency_keywords)
