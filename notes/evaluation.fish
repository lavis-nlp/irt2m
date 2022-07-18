
# RANKING TASK

begin
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-tiny-ranking --source data/evaluation/PSAC-T/2022-06-27_18-07-43
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-small-ranking --source data/evaluation/PSAC-S/2022-06-28_02-41-19
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-medium-ranking --source data/evaluation/PSAC-M/2022-06-28_11-10-17
    irt2m evaluate ranking --irt2 data/irt2/irt2-cde-large-ranking --source data/evaluation/PSAC-L/2022-06-27_18-20-47
end


# LINKING TASK

begin
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-tiny-linking --source data/evaluation/PSAC-T/2022-06-27_18-07-43
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-small-linking --source data/evaluation/PSAC-S/2022-06-28_02-41-19
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-medium-linking --source data/evaluation/PSAC-M/2022-06-28_11-10-17
    irt2m evaluate linking --irt2 data/irt2/irt2-cde-large-linking --source data/evaluation/PSAC-L/2022-06-27_18-20-47
end
