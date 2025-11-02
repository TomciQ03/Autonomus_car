# Autonomus Car — Jupyter (WIP)

Prosty opis pod rekrutację: notatniki i skrypty Pythona do wizji komputerowej (wykrywanie pasów/znaków, sterowanie) w projekcie autonomicznej platformy.

## Jak uruchomić na Windows (bardzo prosto)
1. Zainstaluj Python 3.10+ (podczas instalacji zaznacz **Add Python to PATH**).
2. Otwórz **PowerShell** w folderze projektu (Shift + PPM → „Otwórz w PowerShell”).
3. Utwórz i włącz wirtualne środowisko, zainstaluj paczki:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m ipykernel install --user --name autonomus-car
   ```
4. Uruchom Jupyter Notebook:
   ```powershell
   jupyter notebook
   ```
   Wybierz kernel **autonomus-car**.

## Struktura (przykład)
- `notebooks/` — notatniki `.ipynb`
- `src/` — pomocnicze moduły `.py`
- `data/` — Twoje dane (duże pliki trzymaj lokalnie; nie wrzucaj ich do repo)
- `images/` — zrzuty ekranu/GIF do README
- `requirements.txt` — paczki
- `.gitignore` — co pominąć w repo

## Wskazówki
- Nie wrzucaj ciężkich filmów/danych — dodaj je do `data/` i niech zostaną lokalnie.
- Jeśli brakuje paczki: `pip install NAZWA`, potem dopisz ją do `requirements.txt` i zrób commit.
- To WIP — dopisz krótkie opisy do najważniejszych notatników.

## Licencja
MIT