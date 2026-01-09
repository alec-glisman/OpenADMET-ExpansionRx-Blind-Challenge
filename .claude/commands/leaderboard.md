# Leaderboard Analysis

Scrape and analyze the OpenADMET challenge leaderboard.

## Scrape Leaderboard

```bash
admet leaderboard scrape --user <username> --output results/
```

## Options

```bash
--user <username>     # Your username on leaderboard
--output <dir>        # Output directory for reports
--space <space_id>    # HuggingFace Space (default: openadmet/OpenADMET-ExpansionRx-Challenge)
--no-plots            # Skip generating plots
```

## Output

- CSV files with rankings per endpoint
- Comparison plots
- Summary reports

## ADMET Endpoints (9)

1. LogD
2. Log KSOL (Solubility)
3. Log HLM CLint (Human Liver Metabolism)
4. Log MLM CLint (Mouse Liver Metabolism)
5. Log Caco-2 Papp A>B (Permeability)
6. Log Caco-2 Efflux
7. Log MPPB (Mouse Plasma Protein Binding)
8. Log MBPB (Mouse Blood-Brain)
9. Log MGMB (Mouse Gut-Tissue)

## Key Files

- `src/admet/leaderboard/client.py` - Gradio client
- `src/admet/leaderboard/report.py` - Report generation
