# Отчёт: версионирование данных с DVC и ClearML

## 1. Окружение

| Компонент | Версия |
|---|---|
| Python | 3.12.3 |
| DVC | 3.67.1 |
| ClearML (SDK) | 1.18.0 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |

Версии зафиксированы в `pyproject.toml` / `poetry.lock`.

Git-репозиторий: `<ВСТАВИТЬ ССЫЛКУ НА РЕПОЗИТОРИЙ>`

## 2. DVC

### 2.1. Git-коммиты, теги и состояние данных

| Состояние | Git ref | Commit | md5 `data.dvc` (`.dir`) | Содержимое `data/` |
|---|---|---|---|---|
| base | tag `data-v1` | `a9b5490` | `eb3425236c210bed53327b6c5e01b919` | `iris.csv` |
| wine | ветка `feature/wine` | `96dc468` | `f8de8fd233e4611ca22c4f986e8ba4f3` | `iris.csv`, `wine.csv` |
| iris-features (до merge) | ветка `feature/iris-features` | `aa2a58c` | `ba7c39d9d1ee5015351c2262e24336c8` | `iris.csv` (+2 столбца) |
| итог (merge) | tag `data-v2`, `main` | `f67ba2f` | `d7d9d645aba6d558fac9c42e350ad789` | `iris.csv` (+2 столбца), `wine.csv` |

Граф Git (обе ветки сохранены в истории, у merge-коммита два родителя `aa2a58c` и `96dc468`):

```text
*   f67ba2f (HEAD -> main, tag: data-v2, feature/iris-features) Merge feature/wine into feature/iris-features
|\
| * 96dc468 (feature/wine) data: add wine.csv
* | aa2a58c data: add sepal_area/petal_area to iris
|/
* 7bf9618 Configure DVC local remote (outside repo)
* a9b5490 (tag: data-v1) data v1: base Iris tracked with DVC
```

### 2.2. Хэши

SHA-256 файла `data.dvc` (итоговый, `main`): `c7f33951d287ba426e1fbecd7fee02c7c4279caf6a2f4828645eedfeb0272d92`

| Файл | Версия | SHA-256 | Форма |
|---|---|---|---|
| `iris.csv` | base (`data-v1`, `feature/wine`) | `7d8c276f781bd47bc4abb6c9354cf6262f6d511bacb98cd445c4b8ebee86b5f7` | 150 × 5 |
| `iris.csv` | с признаками (`feature/iris-features`, `data-v2`) | `c7b705193386ac43f2ae64d66952aea19bd11d1d26f5f1f6559e096ae2917911` | 150 × 7 |
| `wine.csv` | `feature/wine`, `data-v2` | `724d80b05e39c7a72fe7631de59583c764049137f9d7089b01b5460e212c75c0` | 178 × 14 |

Проверки: `iris.csv` в `feature/wine` совпадает по SHA-256 с `data-v1`; `wine.csv` в итоговой версии совпадает с `feature/wine`.

### 2.3. DVC remote

Локальная директория `../dvc-remote-storage` (рядом с репозиторием, вне его). В `.dvc/config` лежит относительный путь `../../dvc-remote-storage` (относительно `.dvc/`), секретов нет. После `rm -rf .dvc/cache data` команда `dvc pull` восстановила базовую версию с тем же SHA-256. В конце `dvc status -c` показывает: `Cache and remote 'storage' are in sync`.

### 2.4. Слияние

`git merge feature/wine` в `feature/iris-features` прошёл через DVC merge driver: `Auto-merging data.dvc`, конфликтов нет, маркеров `<<<<<<<` в `data.dvc` — 0. Ручное объединение во временной директории не потребовалось. После `dvc checkout` в workspace два CSV, `check_data.py --expect iris wine` вернул `OK`, затем `dvc push` отправил объект итоговой версии в remote.

### 2.5. Чистое восстановление

`git clone` в новую директорию → `data/` отсутствует (в Git только `data.dvc`) → `dvc pull` (3 файла получены из remote) → `check_data.py --expect iris wine`: `OK`. SHA-256 обоих CSV и md5 `data.dvc` совпали с исходным репозиторием.

## 3. ClearML Dataset

Project: `MIIT/MLOps/practice_3/<фамилия>`, dataset name: `sklearn-tabular-<фамилия>`.

Блок ниже заполняется автоматически командой `python clearml_datasets.py --surname <фамилия> verify`.

<!-- CLEARML:BEGIN -->
_Будет заполнено после запуска `verify`: Dataset IDs (`BASE_ID`, `WINE_ID`, `IRIS_FEATURES_ID`), parent IDs, файлы, added/modified/removed, SHA-256, формы таблиц и dependency graph._
<!-- CLEARML:END -->

## 4. Сравнительная таблица

| Свойство | DVC | ClearML Dataset |
|---|---|---|
| Идентификатор версии | Git commit + md5 `.dir` в `data.dvc` (напр. `data-v2` = `f67ba2f` + `d7d9d645…`) | Dataset ID (32 hex) + строка version (`1.0.0`, `1.1.0`, `2.0.0`); ID берётся из `clearml_ids.json`/отчёта |
| Расположение метаданных | В Git: `data.dvc` (несколько строк: хэш, размер, число файлов, путь) | На ClearML Server (запись Dataset с parents и списком файлов) |
| Расположение файлов | `.dvc/cache` локально + DVC remote `../dvc-remote-storage` (в Git нет ни одного CSV) | ClearML File Server / настроенный storage; локально ClearML cache |
| Создание дочерней версии | Ветка + изменение данных + `dvc add data` + коммит `data.dvc` (`feature/wine`, `feature/iris-features`); `dvc push` | `Dataset.create(..., parent_datasets=[BASE_ID])`, `add_files`, `upload`, `finalize`; в `1.1.0` загружен только `wine.csv` |
| Переключение версии | `git switch` + `dvc checkout`. Без checkout workspace остаётся старым | Нет переключения «на месте»: `Dataset.get(dataset_id=...)` и скачивание в отдельную папку (три версии — три папки) |
| Восстановление на чистой машине | `git clone` + `dvc pull`: `data/` появилась только после pull | `Dataset.get(dataset_id=...)` + `get_mutable_local_copy(target_folder=...)` в пустую папку, Git не нужен |
| Поведение при ветвлении | Git-ветки: две ветки изменили `data.dvc` независимо от `data-v1` | Два ребёнка одного parent: `1.1.0` и `2.0.0` оба от `BASE_ID`, `WINE_ID` не предок `2.0.0` (виден в dependency graph) |
| Поведение при слиянии | `git merge` + DVC merge driver: `Auto-merging data.dvc`, 0 конфликтов, итог — два CSV; merge-коммит с двумя родителями | Автоматического merge нет: нужна новая производная версия (или версия с несколькими parents). В работе версии `1.1.0` и `2.0.0` не сливались |
| Связь версии данных с Git | Жёсткая: версию определяет коммит с `data.dvc`; тег `data-v1`/`data-v2` | Нет автоматической связи; связь только через записанные Dataset IDs в отчёте/коде/конфиге эксперимента |
| Доступность через UI | Сам DVC UI не даёт (в Git-хостинге видно только `data.dvc`; нужен доп. сервис) | ClearML Web UI: список Dataset, версии, теги, lineage-граф |

## 5. Ответы на вопросы

1. **Какой объект однозначно определяет версию данных в DVC?** Хэш содержимого в `data.dvc` (для директории — md5 с суффиксом `.dir`), зафиксированный конкретным Git-коммитом. Например, `data-v1` → `eb342523…`, итог слияния → `d7d9d645…`. Хэш определяет содержимое, а коммит/тег даёт воспроизводимый способ его найти.

2. **Почему одного Git-коммита недостаточно без DVC remote?** В коммите лежит только `data.dvc` с хэшем, а сами байты CSV хранятся в DVC cache и remote. В чистом clone данных нет (`data/` не было до `dvc pull`), и, если объекты не отправлены в remote, получить их неоткуда. В работе после удаления `.dvc/cache` и `data/` версия восстановилась именно из remote.

3. **Чем Dataset version отличается от Dataset ID в ClearML?** Version — человекочитаемая строка (`1.0.0`), которую задаёт автор; она может повториться или быть введена неоднозначно. Dataset ID — уникальный идентификатор конкретной версии на сервере. Поэтому все версии скачивались строго по ID.

4. **Почему `finalize()` нельзя считать созданием следующей версии?** `finalize()` закрывает текущую версию: она становится неизменяемой (Completed), ID остаётся тем же. Следующая версия — это отдельный Dataset с новым ID, создаваемый через `Dataset.create(..., parent_datasets=[...])`. После финализации `1.0.0` потребовалось отдельно создавать `1.1.0` и `2.0.0`.

5. **Как доказать, что ClearML version `2.0.0` создана от базы, а не от `1.1.0`?** Через parent IDs и dependency graph: у `2.0.0` единственный parent равен `BASE_ID`, а `WINE_ID` отсутствует среди предков в `get_dependency_graph()`. Дополнительное подтверждение — в составе `2.0.0` нет `wine.csv`, а `iris.csv` числится как modified относительно базы. Результаты в блоке раздела 3 (проверки «2.0.0 parent == BASE_ID» и «WINE_ID is NOT an ancestor»).

6. **Что произойдёт, если после `git switch` не выполнить DVC checkout?** Git переключит `data.dvc`, но файлы в `data/` останутся от предыдущей версии. В работе после `git switch main` в `data/` остались `iris.csv` и `wine.csv` из `feature/wine`, хотя `data.dvc` уже указывал на базу; `dvc status` показал `modified: data`. Если не заметить, можно обучить модель на данных не той версии или закоммитить неверное состояние. После `dvc checkout` остался только базовый `iris.csv`.

7. **В каком сценарии удобнее DVC, а в каком ClearML Dataset?** DVC удобнее, когда данные должны жить вместе с кодом и ветками: ветки `feature/wine` / `feature/iris-features` и их слияние сделаны обычным Git-процессом, а версия данных воспроизводится по коммиту или тегу. ClearML удобнее, когда версию нужно отдать другой машине или эксперименту по одному ID, видеть lineage в UI и создавать дочерние версии, не перезаливая неизменённые файлы (в `1.1.0` загружен только `wine.csv`).
