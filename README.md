# PyTorch — Programmierpraktikum

Projekt im Rahmen des Programmierpraktikums an der **Universität Osnabrück**.

*TODO: Einzeiler, der das Projekt beschreibt — was geht rein, was kommt raus.*

---

## Inhaltsverzeichnis

- [PyTorch — Programmierpraktikum](#pytorch--programmierpraktikum)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Über das Projekt](#über-das-projekt)
  - [Ziel](#ziel)
  - [Pipeline](#pipeline)
  - [Konzeptioneller Aufbau](#konzeptioneller-aufbau)
  - [Daten](#daten)
  - [Projektstruktur](#projektstruktur)
  - [Voraussetzungen](#voraussetzungen)
  - [Installation](#installation)
    - [1. Conda installieren](#1-conda-installieren)
    - [2. Umgebung anlegen](#2-umgebung-anlegen)
    - [3. Abhängigkeiten installieren](#3-abhängigkeiten-installieren)
    - [4. Installation prüfen](#4-installation-prüfen)
  - [Konfiguration](#konfiguration)
  - [Nutzung](#nutzung)
  - [Reproduzierbarkeit](#reproduzierbarkeit)
  - [Ergebnisse](#ergebnisse)
  - [Mögliche Erweiterungen](#mögliche-erweiterungen)
  - [Arbeitsweise im Team](#arbeitsweise-im-team)
  - [Team](#team)
  - [Credits](#credits)
  - [Lizenz](#lizenz)

---

## Über das Projekt

*TODO: Ausführliche Beschreibung — Problemstellung, Motivation, Abgrenzung.*

## Ziel

*TODO: Was soll am Ende des Praktikums stehen? Woran messen wir, ob es funktioniert hat?*

**Erfolgskriterien**

| Kriterium | Zielwert |
|---|---|
| *TODO* | *TODO* |

## Pipeline

*TODO: Ablauf von den Rohdaten bis zum Ergebnis.*

```
Rohdaten  →  Vorverarbeitung  →  Modell  →  Auswertung
```

*TODO: Diagramm einfügen (z. B. `docs/pipeline.png`).*

## Konzeptioneller Aufbau

*TODO: Architektur genauer — welche Komponenten, welche Modelle, welche Bibliotheken, wie greifen sie ineinander.*

## Daten

| | |
|---|---|
| **Quelle** | *TODO* |
| **Lizenz** | *TODO* |
| **Umfang** | *TODO* |
| **Aufteilung** | *TODO: Train / Validation / Test* |

*TODO: Wie werden die Daten beschafft und vorverarbeitet? Sind sie im Repo oder müssen sie geladen werden?*

## Projektstruktur

```
.
├── main.ipynb               # Einstiegspunkt / Experimente
├── requirements.txt         # Abhängigkeiten
├── Ideas for the Project.md # Ideensammlung
└── README.md
```

*TODO: Aktualisieren, sobald die Struktur steht.*

## Voraussetzungen

| | |
|---|---|
| **Python** | 3.11 |
| **Paketmanager** | conda (empfohlen) oder uv |
| **GPU** | *TODO: erforderlich oder optional?* |
| **Speicherplatz** | *TODO: wie groß werden die Daten?* |

## Installation

Empfohlen wird **conda**, alternativ **uv**.

### 1. Conda installieren

<details open>
<summary><b>macOS</b></summary>

```bash
brew install --cask miniconda
conda init "$(basename "$SHELL")"
```

Danach das Terminal einmal neu starten.

</details>

<details>
<summary><b>Windows</b></summary>

```powershell
winget install Anaconda.Miniconda3
```

> [!IMPORTANT]
> Nach der Installation ist `conda` in einer normalen PowerShell **nicht** verfügbar.
> Es gibt zwei Wege:
>
> **a)** Die **Anaconda Prompt** aus dem Startmenü benutzen — funktioniert sofort.
>
> **b)** PowerShell einmalig einrichten:
>
> ```powershell
> conda init powershell
> ```
>
> Falls das Aktivieren danach mit einer Fehlermeldung zur Ausführungsrichtlinie
> abbricht, einmalig freigeben und PowerShell neu starten:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

</details>

### 2. Umgebung anlegen

```bash
conda create --name pytorch-projekt python=3.11
conda activate pytorch-projekt
```

### 3. Abhängigkeiten installieren

```bash
conda install --file requirements.txt
```

> [!NOTE]
> Die passende PyTorch-Variante hängt von Betriebssystem und GPU ab (CUDA, MPS oder CPU).
> Den passenden Befehl gibt der offizielle Konfigurator aus: https://pytorch.org/get-started/locally/

<details>
<summary>Alternative mit uv</summary>

```bash
uv venv --python 3.11
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

</details>

### 4. Installation prüfen

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Konfiguration

*TODO: Werden API-Schlüssel oder Zugangsdaten benötigt?*

Zugangsdaten gehören in eine lokale `.env`-Datei und **niemals ins Repository**.
Als Vorlage dient `.env.example`:

```bash
cp .env.example .env
```

## Nutzung

*TODO: Wie startet man das Projekt? Beispielaufruf, erwartete Ausgabe.*

```bash
# TODO
```

## Reproduzierbarkeit

*TODO: Welcher Seed wird gesetzt? Wo liegen Konfiguration und Modellgewichte?*

*TODO: Welche Hardware wurde für die berichteten Ergebnisse verwendet?*

## Ergebnisse

*TODO: Metriken, Beispielausgaben, Grafiken.*

## Mögliche Erweiterungen

*TODO: Was wurde bewusst vereinfacht?*

*TODO: Was wäre der nächste Schritt, wenn mehr Zeit da wäre?*

*TODO: Verbesserungsmöglichkeiten an genutzten Fremd-Repositories.*

## Arbeitsweise im Team

*TODO: An Team anpassen.*

- Entwickelt wird in Feature-Branches, `main` bleibt lauffähig
- Vor dem Push: `git pull --rebase origin main`, damit die Historie linear bleibt
- Notebooks vor dem Commit ausführen und Ausgaben leeren, sonst gibt es unnötige Konflikte

## Team

| GitHub | Name | Rolle |
|---|---|---|
| [@nbomers](https://github.com/nbomers) | *TODO* | *TODO* |
| [@D4ne2kk](https://github.com/D4ne2kk) | *TODO* | *TODO* |
| [@eknight04](https://github.com/eknight04) | *TODO* | *TODO* |

## Credits

**Genutzte Repositories und Datensätze**

- *TODO: Repository / Datensatz — Lizenz — wofür verwendet*

**Werkzeuge**

- [claude.ai](https://claude.ai) — Unterstützung bei Recherche und Entwicklung

**Literatur**

- *TODO*

## Lizenz

*TODO: Lizenz festlegen und `LICENSE`-Datei ergänzen — oder vermerken, dass das Projekt nicht zur Weiterverwendung freigegeben ist.*
