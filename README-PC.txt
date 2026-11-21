CFB CLIPPER - PC PAR CHALANE KA TAREEQA
=========================================

1. Python 3.12 install karo: https://www.python.org/downloads/
   Install karte waqt "Add Python to PATH" zaroor tick karna.

2. FFmpeg install karo: PowerShell kholo, ye likho:
   winget install Gyan.FFmpeg

3. Is zip ko extract karo, jaise: C:\cfb-clipper

4. start.bat par double-click karo.
   PEHLI BAAR packages download honge (10-15 min, internet chahiye).
   Agl i baar se foran khul jayega.

5. Browser khud khul jayega: http://127.0.0.1:8000
   Wahan YouTube link ya video file do, clip banao, download karo.

6. Band karne ke liye: kaali wali window band kar do.

NOTE:
- Tesseract optional hai (sirf A/V sync ke liye). Na ho to bhi clips bante hain.
- Pehli render par Whisper model download hoga (~500MB), aik dafa hi.
- .env file is zip me NAHI hai (us me secrets hote hain). Zaroorat pare to
  .env.example ko copy karke .env banao.
