library(lme4)
library(car)
library(survival)
library(coxme)
library(simr)
library(ggplot2)

# parse command line arguments
args = commandArgs(trailingOnly=TRUE)
mlm.in <- args[1]
cox.in <- args[2]
wls.in <- args[3]
wls_plot.out <- args[4]

data = read.csv(mlm.in)
data_cox = read.csv(cox.in)
data_wls_cent = read.csv(wls.in)

# centre level
data_wls_cent$Count = data_wls_cent$TTR.count
head(data_wls_cent)

# perform weighted least squares regression
wls_model_cent = lm(TTR.mean ~ algorithm_consistency.mean, data = data_wls_cent, weights = data_wls_cent$Count)

# view summary of model
summary(wls_model_cent)

p <- (
    ggplot(data_wls_cent, aes(x = algorithm_consistency.mean, y = TTR.mean, size = Count)) + 
    geom_point(shape = 21) + 
    geom_smooth(method = "lm", mapping = aes(weight = TTR.count), color = "black", show.legend = FALSE) + 
    ggtitle("Weighted Least Square Model (Threshold Method)") + 
    xlab("Mean centre algorithm-consistency") + 
    ylab("Mean centre TTR")
)

ggsave(wls_plot.out)

data['centre_algorithm_consistency'] = data['centre_algorithm_consistency']*10
data['TTR'] = data['TTR']*100

fit_ac = lmer(TTR ~ 
              AGE + 
              WT + 
              male + 
              white + 
              smoker + 
              hf + 
              hypt + 
              diab + 
              stroke + 
              warfuse + 
              amiod + 
              insulin + 
              centre_algorithm_consistency + 
              (1|CENTID) + 
              sechosp + # centre level
              acclinic + # centre level
              (1|CTRYID) + 
              highincome + # ctry level
              dale + # ctry level
              hsp, data = data)

summary(fit_ac)

Anova(fit_ac)

confint(fit_ac, level = 0.95)

time = as.numeric(data_cox$time_to_event)
status = as.numeric(data_cox$composite_outcome)
data_cox['centre_algorithm_consistency'] = data_cox['centre_algorithm_consistency']*10

fit_coxme = coxme(
    Surv(time, status) ~ 
    AGE + 
    WT + 
    male + 
    white + 
    smoker + 
    hf + 
    hypt + 
    diab + 
    stroke + 
    warfuse + 
    amiod + 
    insulin + 
    beta_blocker + # medication
    asa + # medication
    ace + # medication
    statin + # medication
    (1|CENTID) + 
    centre_algorithm_consistency + 
    sechosp + # centre level
    acclinic + # centre level
    (1|CTRYID) + 
    highincome + # ctry level
    dale + # ctry level
    hsp, # ctry level
    data = data_cox)

summary(fit_coxme)

exp(confint(fit_coxme, level = 0.95))
