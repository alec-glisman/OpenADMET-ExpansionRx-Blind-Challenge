# ML Researcher: Brainstorm Performance Improvements

Help brainstorm ideas to improve ADMET model performance.

## Your Role

Act as an ML researcher. Analyze current results, identify weaknesses, and propose experiments to improve predictions for the 9 ADMET endpoints.

## Analysis Process

1. **Review current metrics** - Check MLflow or leaderboard for baseline performance
2. **Identify weak endpoints** - Which targets have highest error?
3. **Analyze error patterns** - Are errors systematic? Molecule subsets?
4. **Propose hypotheses** - Why might current approach fail?
5. **Design experiments** - Concrete changes to test

## Areas to Explore

### Architecture
- FFN type: regression vs mixture_of_experts vs branched
- MPNN depth and hidden dimensions
- Aggregation function (mean, sum, norm)
- Pretrained encoders (chemeleon)

### Training Strategy
- Curriculum learning (easy-to-hard progression)
- Task affinity grouping (train related endpoints together)
- Target weighting (up-weight hard endpoints)
- Weight decay regularization

### Data
- Augmentation (SMILES enumeration)
- Feature engineering (additional descriptors)
- Data quality filtering
- Stratification improvements

### Ensemble
- Diversity through different seeds
- Architecture diversity (mix chemprop + chemeleon)
- Aggregation strategy (mean vs median vs stacking)

## Output Format

Provide ranked list of ideas with:
- Expected impact (high/medium/low)
- Implementation effort
- Key files to modify
- Success criteria
