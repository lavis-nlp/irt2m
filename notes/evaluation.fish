
# RANKING TASK

begin

    # PSAC
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-tiny-ranking --source data/evaluation/PSAC-T/2022-06-27_18-07-43
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-small-ranking --source data/evaluation/PSAC-S/2022-06-28_02-41-19
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-medium-ranking --source data/evaluation/PSAC-M/2022-06-28_11-10-17
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-large-ranking --source data/evaluation/PSAC-L/2022-06-27_18-20-47

    # PMAC
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-tiny-ranking --source data/evaluation/PMAC-T/2022-06-28_09-54-58
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-small-ranking --source data/evaluation/PMAC-S/2022-06-28_14-08-31
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-medium-ranking --source data/evaluation/PMAC-M/2022-06-28_23-47-21
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-large-ranking --source data/evaluation/PMAC-L/2022-06-29_14-30-56

    # JSC
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-tiny-ranking --source data/evaluation/JSC-T/2022-07-15_16-28-15
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-small-ranking --source data/evaluation/JSC-S/2022-07-18_01-46-16
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-medium-ranking --source data/evaluation/JSC-M/2022-07-17_06-15-30
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-large-ranking --source data/evaluation/JSC-L/2022-07-15_12-03-25


end


# LINKING TASK

begin

    # PSAC
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-tiny-linking --source data/evaluation/PSAC-T/2022-06-27_18-07-43
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-small-linking --source data/evaluation/PSAC-S/2022-06-28_02-41-19
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-medium-linking --source data/evaluation/PSAC-M/2022-06-28_11-10-17
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-large-linking --source data/evaluation/PSAC-L/2022-06-27_18-20-47

    # PMAC
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-tiny-linking --source data/evaluation/PMAC-T/2022-06-28_09-52-54
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-small-linking --source data/evaluation/PMAC-S/2022-07-11_17-47-24
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-medium-linking --source data/evaluation/PMAC-M/2022-06-28_14-35-23
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-large-linking --source data/evaluation/PMAC-L/2022-06-29_14-30-56

    # JSC
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-tiny-linking --source data/evaluation/JSC-T/2022-07-15_11-51-35
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-small-linking --source data/evaluation/JSC-S/2022-07-18_09-37-00
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-medium-linking --source data/evaluation/JSC-M/2022-07-17_06-15-30
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-large-linking --source data/evaluation/JSC-L/2022-07-15_12-03-25

    # JMC
    JMC-T/2022-07-21_22-28-01

end
