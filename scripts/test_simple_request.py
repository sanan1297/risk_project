"""Test single request to SECOP."""
import requests, math, re, time

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"
r = s.get(url, timeout=15)
print(f"Status: {r.status_code}, Len: {len(r.text)}")

if "__zjc" in r.text:
    print("Zenedge challenge detected")
    m = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
    if m:
        v = math.floor(int(m.group(1)) * float(m.group(2)))
        ck = re.search(r'cookie = "(\w+)="', r.text)
        if ck:
            s.cookies.set(ck.group(1), str(v), domain="www.contratos.gov.co")
            rd = re.search(r"window\.location='([^']+)'", r.text)
            if rd:
                ru = rd.group(1)
                if not ru.startswith("http"):
                    ru = "https://www.contratos.gov.co" + ru
                time.sleep(0.2)
                r = s.get(ru, timeout=15)
                print(f"After redirect: Status={r.status_code}, Len={len(r.text)}")

upper = r.text.upper()
print(f"MATRIZ in text: {'MATRIZ' in upper}")
print(f"RIESGO in text: {'RIESGO' in upper}")
print(f"ESTUDIOS PREVIOS in text: {'ESTUDIOS PREVIOS' in upper}")
print(f"Body length: {len(r.text)}")
