// predict.js: applies the exported logistic regression in the browser.
// Scaling is (value - mean) / std for each numeric attribute, then scores = intercept + coefficients . z,
// then softmax gives the class probabilities. This matches Spark's multinomial logistic regression.
(function (global) {
  function softmax(scores) {
    const m = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - m));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / total);
  }

  function predict(model, values) {
    const z = model.features.map((name, i) => {
      const std = model.stds[i] || 1;
      return (values[name] - model.means[i]) / std;
    });
    const scores = model.classes.map((_, k) =>
      model.intercepts[k] + model.coefficients[k].reduce((acc, w, i) => acc + w * z[i], 0));
    const probs = softmax(scores);
    return model.classes.map((label, k) => ({ label: label, probability: probs[k] }));
  }

  const api = { predict: predict, softmax: softmax };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.StressModel = api;
})(typeof window !== "undefined" ? window : globalThis);
