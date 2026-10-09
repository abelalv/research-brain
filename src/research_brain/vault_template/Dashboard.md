# Dashboard

> [!info] Requires the **Dataview** community plugin (Settings → Community plugins → Browse → Dataview).

## Ideas to develop
```dataview
TABLE operator, novelty, feasibility, impact, status
FROM "50-ideas"
WHERE status != "dropped"
SORT (novelty + feasibility + impact) DESC
```

## Arguments that need work
```dataview
TABLE verdict, repaired, summary
FROM "60-arguments"
WHERE verdict != "valid" OR repaired = true
SORT updated DESC
```

## Low-confidence or open atoms
```dataview
TABLE kind, confidence, source
FROM "30-atoms"
WHERE confidence = "low" OR kind = "question" OR kind = "hypothesis"
SORT updated DESC
LIMIT 30
```

## Atoms by kind
```dataview
TABLE length(rows) AS count
FROM "30-atoms"
GROUP BY kind
```

## Concepts (most recently updated)
```dataview
TABLE level, summary
FROM "40-concepts"
SORT updated DESC
LIMIT 20
```

## Recent sources
```dataview
TABLE summary
FROM "20-summaries"
SORT created DESC
LIMIT 10
```
