at::Tensor gaussian_probability(at::Tensor &sigma, at::Tensor &mu, 
    at::Tensor &data) {
    data = data.toType(at::kDouble);
    sigma = sigma.toType(at::kDouble);
    mu = mu.toType(at::kDouble);
    data = data.unsqueeze(1).expand_as(sigma);
    auto exponent = -0.5 * at::pow((data - mu) / sigma, at::Scalar(2));
    auto ret = ONEOVERSQRT2PI * (exponent.exp() / sigma);
    return at::prod(ret, 2);
}

at::Tensor mdn_accuracy(at::Tensor &pi, at::Tensor &sigma, 
    at::Tensor &mu, at::Tensor &target) {
    auto prob_double = pi * gaussian_probability(sigma, mu, target);
    auto prob_float = prob_double.toType(at::kFloat);
    auto safe_sum = at::add(at::sum(prob_float, at::IntList(1)), at::Scalar(0.000001));
    return at::mean(safe_sum);
}
