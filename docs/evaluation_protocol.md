# Evaluation protocol

This document describes the supporting evaluator's behavior. It is not a report of a new ACE-MG experiment.

## Query subsets for F1

For an episode with Novel class N, Unknown class U and three Base classes B:

- Base Macro-F1 uses queries whose true label is in B. It computes the per-class F1 values for all three Base labels and averages them equally.
- Novel F1 uses queries whose true label is in B or N and evaluates N as the positive class.
- K=0 uses the highest-scoring enrolled-class prediction; K≥1 uses the saved prediction that includes the Reject option.

The query restriction precedes counting true positives, false positives and false negatives. It changes the result compared with calculating those counts over all queries.

For example, suppose each Base class has one correctly classified query, and one Novel query is incorrectly assigned to a Base class. Base-only Macro-F1 remains 1.0. A Base-label Macro-F1 evaluated on all enrolled queries would instead be 8/9. Similarly, an Unknown query incorrectly assigned to Novel does not enter this evaluator's known-query Novel F1, but would count as a false positive under an all-query definition.

These are conditional recognition metrics. Unknown-positive AUROC/AP and OSCR separately measure unknown discrimination and joint open-set behavior. Conditional F1 values must not be described as F1 over the full open-set query population.

## Unknown metrics

The saved knownness score is the highest final enrolled-class score minus the Reject score. The evaluator cannot establish that an input score was produced by the manuscript's method.

- Unknown AUROC uses negative knownness and treats Unknown queries as positive. Score ties receive half credit.
- Unknown AUPR is average precision, with tied scores processed as one threshold group.
- OSCR sweeps knownness. CCR counts correctly classified enrolled queries accepted above the threshold, divided by all enrolled queries. FPR counts accepted Unknown queries, divided by all Unknown queries. Correctness is based on the closed enrolled-class prediction, not the fixed Reject decision. The CCR–FPR curve is integrated using trapezoids with score ties handled together.

## Aggregation

For each fold, K and actual ordered Novel/Unknown pair, metrics are averaged across support draws. Role means are averaged equally to obtain a fold mean. Fold means are averaged equally, with sample standard deviation across folds (`ddof=1`). In the complete protocol there are five folds, 12 actual class-role pairs, one K=0 episode per pair, and 20 support draws for each K>0, yielding 3,660 episodes in total.

The strict protocol checks the supplied episode structure and class coverage. At K≥1, a positive knownness score requires the closed enrolled-class prediction to be the open prediction; a negative score requires Reject. A zero score permits either. At K=0 this Reject-consistency rule is not applied because hard rejection is disabled. A custom protocol cannot change the manuscript's fold, K or seed sets while retaining strict mode.

These checks do not inspect training data, support identities, source recordings or session manifests, and do not establish the authenticity of the score values.

## Relationship to the manuscript

The manuscript's Normal-fixed-Base protocol, K settings, K-dependent Reject rule, unknownness direction, OSCR construction, candidate-set shift regularization λ=1000, graph propagation α=0.35 with 8 iterations, and hierarchical mean/sample-SD aggregation correspond to this package's documented settings.

The manuscript description must explicitly state the F1 query subsets if its reported tables were computed using this evaluator's definitions. If the original results used a different query population, the appropriate evaluator must be used instead. No unit test or synthetic example can establish which definition generated a historical result table.

The code package does not include real predictions or historical summaries. It makes no claim of independent numerical reproduction of the manuscript tables.
