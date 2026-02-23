// Single-level Logistic Regression without imputation
data {
  int<lower=1> N;                  // number of complete case samples
  int<lower=1> D;                  // number of predictors
  matrix[N, D] DESIGN;             // design matrix (only complete cases)
  int<lower=0, upper=1> OUTCOME[N]; // binary outcome for complete cases

  // Prediction data (remains the same)
  int<lower=1> PRED_N;
  matrix[PRED_N, D] PRED_DESIGN;
  real<lower=0> scale_global;
}

parameters {
  real alpha;                      // overall intercept
  vector[D] beta;                  // fixed effects
  real<lower=0> tau;               // scale parameter for the Laplace prior on beta
}

transformed parameters {
  // Calculate log-odds using only fixed effects and complete case data
  vector[N] logodds = alpha + DESIGN * beta;
}

model {
  // Priors
  alpha ~ cauchy(0, 2.5);
  tau ~ cauchy(0, scale_global);
  beta ~ double_exponential(0, tau);
  

  // Likelihood using complete case data
  OUTCOME ~ bernoulli_logit(logodds);
}

generated quantities {
  array[PRED_N] real pred;
  vector[PRED_N] pred_logodds;

  // Calculate predictions using only fixed effects
  for (n in 1:PRED_N) {
    pred_logodds[n] = alpha + dot_product(PRED_DESIGN[n], beta);
    // Convert log-odds to probability
    pred[n] = inv_logit(pred_logodds[n]);
  }
}
