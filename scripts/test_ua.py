"""Test request with proper User-Agent."""
import requests

# Test 1: proper UA
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"
r = s.get(url, timeout=15)
print(f"Test 1 (full UA): Status={r.status_code}, Len={len(r.text)}")

# Test 2: minimal UA  
s2 = requests.Session()
s2.headers.update({"User-Agent": "Mozilla/5.0"})
r2 = s2.get(url, timeout=15)
print(f"Test 2 (minimal UA): Status={r2.status_code}, Len={len(r2.text)}")

# Check each
for i, (label, resp) in enumerate([("Test 1", r), ("Test 2", r2)], 1):
    upper = resp.text.upper()
    print(f"\n{label}:")
    print(f"  MATRIZ: {'MATRIZ' in upper}")
    print(f"  RIESGO: {'RIESGO' in upper}")
    print(f"  ESTUDIOS PREVIOS: {'ESTUDIOS PREVIOS' in upper}")
    print(f"  First 200 chars: {resp.text[:200]}")
