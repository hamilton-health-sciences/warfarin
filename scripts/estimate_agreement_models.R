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
ttr_table.out <- args[5]
events_table.out <- args[6]
method.name <- args[7]

if (mlm.in != "none") data = read.csv(mlm.in)
data_cox = read.csv(cox.in)
if (wls.in != "none") data_wls_cent = read.csv(wls.in)

# centre level
if (wls.in != "none") {
    data_wls_cent$Count = data_wls_cent$TTR.count
    head(data_wls_cent)

    # perform weighted least squares regression
    wls_model_cent = lm(TTR.mean ~ algorithm_consistency.mean,
                        data = data_wls_cent, weights = data_wls_cent$Count)

    # view summary of model
    print(summary(wls_model_cent))

    p <- (
        ggplot(data_wls_cent, aes(x = algorithm_consistency.mean, y = TTR.mean, size = Count)) + 
        geom_point(shape = 21) + 
        geom_smooth(method = "lm", mapping = aes(weight = TTR.count), color = "black", show.legend = FALSE) + 
        ggtitle(paste("Weighted Least Square Model (", method.name, " Method)", sep="")) +
        xlab("Mean centre algorithm-consistency") + 
        ylab("Mean centre TTR") +
        xlim(c(0, 1))
    )

    ggsave(wls_plot.out)
}

if (mlm.in != "none") {
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

    anova.ttr <- Anova(fit_ac)
    anova.ttr

    confint.ttr <- confint(fit_ac, level = 0.95)
    confint.ttr
}

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

cox.summary <- summary(fit_coxme)
cox.summary

confint.events <- exp(confint(fit_coxme, level = 0.95))
confint.events

if (mlm.in != "none") {
    # Format for output
    varnames <- rownames(anova.ttr)
    coef.ttr <- summary(fit_ac)$coef[varnames, "Estimate"]
    coef.ttr.fmt <- format(round(coef.ttr, 2), nsmall=2, trim=TRUE)
    ci.lower <- confint.ttr[names(coef.ttr), 1]
    ci.lower.fmt <- format(round(ci.lower, 2), nsmall=2, trim=TRUE)
    ci.upper <- confint.ttr[names(coef.ttr), 2]
    ci.upper.fmt <- format(round(ci.upper, 2), nsmall=2, trim=TRUE)
    pval.fmt <- format(round(anova.ttr$`Pr(>Chisq)`, 3), nsmall=3, trim=TRUE)
    df.ttr <- data.frame(
        coefficient=coef.ttr.fmt,
        ci=paste(ci.lower.fmt, ci.upper.fmt, sep=", "),
        pval=pval.fmt
    )

    write.table(df.ttr, file=ttr_table.out, sep=",")
}

extract_coxme_table <- function(mod) {
    # Taken from: https://stackoverflow.com/questions/43720260/how-to-extract-p-values-from-lmekin-objects-in-coxme-package
    beta <- mod$coefficients
    nvar <- length(beta)
    nfrail <- nrow(mod$var) - nvar
    se <- sqrt(diag(mod$var)[nfrail + 1:nvar])
    z<- round(beta/se, 2)
    p<- signif(1 - pchisq((beta/se)^2, 1), 2)
    table=data.frame(cbind(beta,se,z,p))

    return(table)
}

varnames <- names(cox.summary$coefficients)
coef.events <- exp(cox.summary$coefficients)
coef.events.fmt <- format(round(coef.events, 2), nsmall=2, trim=TRUE)
ci.lower <- confint.events[names(coef.events), 1]
ci.lower.fmt <- format(round(ci.lower, 2), nsmall=2, trim=TRUE)
ci.upper <- confint.events[names(coef.events), 2]
ci.upper.fmt <- format(round(ci.upper, 2), nsmall=2, trim=TRUE)
pval <- extract_coxme_table(fit_coxme)$p
pval.fmt <- format(round(pval, 3), nsmall=3, trim=TRUE)
df.events <- data.frame(
    hr=coef.events.fmt,
    ci=paste(ci.lower.fmt, ci.upper.fmt, sep=", "),
    pval=pval.fmt
)

write.table(df.events, file=events_table.out, sep=",")
