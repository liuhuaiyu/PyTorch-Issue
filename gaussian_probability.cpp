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
