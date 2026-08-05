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
  - [Installation](#installation)
    - [1. Conda installieren](#1-conda-installieren)
    - [2. Umgebung anlegen](#2-umgebung-anlegen)
    - [3. Abhängigkeiten installieren](#3-abhängigkeiten-installieren)
  - [Nutzung](#nutzung)
  - [Ergebnisse](#ergebnisse)
  - [Mögliche Erweiterungen](#mögliche-erweiterungen)
  - [Team](#team)
  - [Credits](#credits)

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
├── main.ipynb              # Einstiegspunkt / Experimente
├── requirements.txt        # Abhängigkeiten
├── Ideas for the Project.md # Ideensammlung
└── README.md
```

*TODO: Aktualisieren, sobald die Struktur steht.*

## Installation

Empfohlen wird **conda**, alternativ **uv**.

### 1. Conda installieren

**macOS**

```bash
brew install --cask miniconda
```

**Windows**

```powershell
winget install Anaconda.Miniconda3
```

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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

</details>

## Nutzung

*TODO: Wie startet man das **Projekt**? Beispielaufruf, erwartete Ausgabe.*

```bash
# TODO
```

## Ergebnisse

*TODO: Metriken, Beispielausgaben, Grafiken.*

## Mögliche Erweiterungen

*TODO: Was wurde bewusst vereinfacht?*

*TODO: Was wäre der nächste Schritt, wenn mehr Zeit da wäre?*

*TODO: Verbesserungsmöglichkeiten an genutzten Fremd-Repositories.*

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
