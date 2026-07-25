import numpy as np

from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class BaseRho:
    """
    Base class for robust loss functions.
    """

    def rho(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def psi(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TukeyBisquare(BaseRho):
    def __init__(self, c: float = 1.56):
        """
        Tukey's bisquare loss function.

        Args:
            c (float, optional):
                Tuning constant controlling the cutoff. Defaults to 1.56.
        """
        self.c = c

    def rho(self, X: np.ndarray) -> np.ndarray:
        return np.where(
            np.abs(X) <= self.c,
            (X**2 / 2) - (X**4 / (2 * self.c**2)) + (X**6 / (6 * self.c**4)),
            self.c**2 / 6,
        )

    def psi(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.c, X * (1 - (X / self.c) ** 2) ** 2, 0)


class Huber(BaseRho):
    def __init__(self, b: float = 1.5):
        """
        Huber's loss function.

        Args:
            b (float, optional):
                Threshold between the quadratic and linear regions of loss. Defaults to 1.5.
        """
        self.b = b

    def rho(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.b, X**2 / 2, self.b * np.abs(X) - self.b**2 / 2)

    def psi(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.b, X, self.b * np.sign(X))


def _psi_wrap_scalar(y: float, b: float, c: float, A: float, B: float, k: float) -> float:
    """Wrapping psi function used internally by :func:`calculate_q1q2`.

    Port of ``psiWrap`` in the R cellPCA code.
    """
    ay = abs(y)
    if ay >= c:
        return 0.0
    if ay >= b:
        return (
            np.sqrt(A * (k - 1))
            * np.tanh(0.5 * np.sqrt((k - 1) * B**2 / A) * (c - ay))
            * np.sign(y)
        )
    return y


def calculate_q1q2(
    b: float, c: float, maxit: int = 500, prec_scale: float = 1e-10
) -> tuple[float, float]:
    """Compute the tuning constants ``q1`` and ``q2`` that make the wrapping psi function
    continuous for corner point ``b`` and rejection point ``c``.

    Port of ``calculateq1q2`` in the R cellPCA code.

    Args:
        b (float): corner point of the wrapping psi function.
        c (float): rejection point of the wrapping psi function.
        maxit (int, optional): maximum number of iterations. Defaults to 500.
        prec_scale (float, optional): convergence tolerance. Defaults to 1e-10.

    Returns:
        tuple[float, float]: the constants ``(q1, q2)``.
    """
    A = 2 * norm.cdf(c) - 1 - 2 * c * norm.pdf(c)
    B = 2 * norm.cdf(c) - 1
    k = max(1.0, c)
    Anew, Bnew, knew = A, B, k
    for _ in range(maxit):
        A, B, k = Anew, Bnew, knew
        res = minimize_scalar(
            lambda y: abs(
                b - np.sqrt(A * (y - 1)) * np.tanh(0.5 * np.sqrt((y - 1) * B**2 / A) * (c - b))
            ),
            bounds=(1 + prec_scale, 1000),
            method="bounded",
            options={"xatol": prec_scale},
        )
        knew = res.x
        Anew = quad(lambda y: _psi_wrap_scalar(y, b, c, A, B, k) ** 2 * norm.pdf(y), -c, c)[0]
        Bnew = quad(
            lambda y: abs(_psi_wrap_scalar(y, b, c, A, B, k)) * abs(y) * norm.pdf(y), -c, c
        )[0]
        if max(abs(Anew - A), abs(Bnew - B), abs(knew - k)) < prec_scale:
            break
    q1 = np.sqrt(Anew * (knew - 1))
    q2 = Bnew / 2 * np.sqrt((knew - 1) / Anew)
    return q1, q2


class TanhWrapping(BaseRho):
    """Hyperbolic-tangent wrapping rho function used by cellPCA and MacroPCA.

    This is the rho/weight family underlying the wrapping M-estimator of
    Raymaekers & Rousseeuw (2021). It exposes two parametrizations, matching the
    R functions ``rhoTanh`` / ``weightTanh``:

    - The default parametrization (``bb``/``cc`` left as ``None``) rescales the
      input by ``1.5 / b`` and uses the fixed constants ``c = 4``,
      ``q1 = 1.540793``, ``q2 = 0.8622731``. This is the cellwise rho (rho_1).
    - The custom parametrization (both ``bb`` and ``cc`` given) uses arbitrary
      corner/rejection points, computing ``q1``/``q2`` via :func:`calculate_q1q2`
      when they are not supplied. This is used for the casewise rho (rho_2).

    Args:
        b (float, optional): tuning constant of the default parametrization.
            Defaults to 1.5.
        bb (float | None, optional): corner point of the custom parametrization.
        cc (float | None, optional): rejection point of the custom parametrization.
        q1 (float | None, optional): continuity constant (computed when None).
        q2 (float | None, optional): continuity constant (computed when None).

    References:
        - Raymaekers, J., & Rousseeuw, P. J. (2021). Fast robust correlation for
          high-dimensional data. Technometrics, 63(2), 184-198.
    """

    def __init__(
        self,
        b: float = 1.5,
        *,
        bb: float | None = None,
        cc: float | None = None,
        q1: float | None = None,
        q2: float | None = None,
    ):
        if bb is None or cc is None:
            # default parametrization (rho_1)
            self.default = True
            self.b = b
            self.bb = 1.5
            self.cc = 4.0
            self.q1 = 1.540793
            self.q2 = 0.8622731
            self.dd = 3.7622126668
        else:
            # custom parametrization (rho_2)
            self.default = False
            self.b = b
            self.bb = bb
            self.cc = cc
            self.quadratic = bb > 100
            if not self.quadratic:
                if q1 is None or q2 is None:
                    q1, q2 = calculate_q1q2(bb, cc)
                self.q1 = q1
                self.q2 = q2
                self.dd = bb**2 / 2 + (q1 / q2) * np.log(np.cosh(q2 * (cc - bb)))

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return 1.5 * X / self.b if self.default else X

    def rho(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.default and self.quadratic:
            return X**2
        xx = self._standardize(X)
        a = np.abs(xx)
        q12 = self.q1 / self.q2
        out = np.full(xx.shape, np.nan)
        small = a <= self.bb
        mid = (a > self.bb) & (a < self.cc)
        big = a >= self.cc
        out[small] = xx[small] ** 2 / 2
        out[mid] = self.dd - q12 * np.log(np.cosh(self.q2 * (self.cc - a[mid])))
        out[big] = self.dd
        return out

    def weight(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.default and self.quadratic:
            out = np.ones(X.shape)
            out[np.isnan(X)] = np.nan
            return out
        xx = self._standardize(X)
        a = np.abs(xx)
        psi = np.full(xx.shape, np.nan)
        small = a <= self.bb
        mid = (a > self.bb) & (a <= self.cc)
        big = a > self.cc
        psi[small] = xx[small]
        psi[mid] = self.q1 * np.tanh(self.q2 * (self.cc - a[mid])) * np.sign(xx[mid])
        psi[big] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = psi / xx
        tt[xx == 0] = 1.0
        return tt

    def psi(self, X: np.ndarray) -> np.ndarray:
        return self.weight(X) * np.asarray(X, dtype=float)


def get_c_scale_tanh(delta: float = 1.8811) -> float:
    """Calibration constant for :func:`scale_tanh` for a given ``delta``.

    Port of ``getCScaleTanh`` in the R cellPCA code. For the default
    ``delta = 1.8811`` the calibrated value ``0.34308`` is returned directly.
    """
    if delta == 1.8811:
        return 0.34308
    rng = np.random.default_rng(123)
    x = rng.standard_normal(5000)
    rho = TanhWrapping(b=1.5)
    c_vec = np.arange(0.1, 10 + 1e-9, 0.0001)
    m_vec = np.array([np.mean(rho.rho(x / c)) for c in c_vec])
    return float(c_vec[np.argmin(np.abs(m_vec - delta))])


def scale_tanh(
    u: np.ndarray,
    delta: float = 1.8811,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Hyperbolic-tangent M-scale estimator.

    Port of ``scaleTanh`` in the R cellPCA code. NaN values are ignored.

    Args:
        u (np.ndarray): univariate data.
        delta (float, optional): consistency target. Defaults to 1.8811.
        max_iter (int, optional): maximum number of iterations. Defaults to 100.
        tol (float, optional): convergence tolerance. Defaults to 1e-6.

    Returns:
        float: the estimated scale (0.0 if the data has essentially no spread).
    """
    u = np.asarray(u, dtype=float)
    u = u[~np.isnan(u)]
    c = get_c_scale_tanh(delta)
    rho = TanhWrapping(b=1.5)
    s0 = np.median(np.abs(u)) / 0.6745
    if s0 < np.finfo(float).eps:
        return 0.0
    err = tol + 1
    it = 0
    while err > tol and it < max_iter:
        it += 1
        s1 = np.sqrt(s0**2 * np.mean(rho.rho(u / (c * s0))) / delta)
        err = abs(s1 - s0) / s0
        s0 = s1
    return float(s0)
