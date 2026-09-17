"""
One-off diagnostic: lists every model your Gemini API key currently has
access to, so we stop guessing model name strings that keep getting
deprecated out from under us. Run this once, paste the output back.
"""
from google import genai

client = genai.Client()

print("Models available to your key that support generateContent:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
    if not actions or "generateContent" in actions or any("generateContent" in str(a) for a in actions):
        print(f"  {m.name}")
