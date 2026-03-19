"""
Math Visualization Modules Registry.

Defines the available interactive math visualizations that the frontend can render.
Each module describes its concept, parameters,  and the data the backend can compute.
The frontend uses Three.js / Canvas to render; the backend provides evaluation endpoints.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

VISUALIZATION_MODULES = {
    # ── Geometry ──────────────────────────────────────────────────────────
    "unit_circle": {
        "name": "Unit Circle Explorer",
        "category": "Geometry",
        "icon": "📐",
        "description": "Animated unit circle with draggable angle point, live sin/cos/tan readout",
        "params": {"angle": {"type": "slider", "min": 0, "max": 360, "step": 1, "default": 45, "unit": "degrees"}},
        "outputs": ["sin", "cos", "tan", "x", "y", "radians"],
        "curriculum_topics": ["The unit circle", "Sine, cosine, tangent definitions"],
    },
    "triangle_solver": {
        "name": "Triangle Solver",
        "category": "Geometry",
        "icon": "📐",
        "description": "Input any 2 sides/angles, auto-solve the rest with color-coded sides",
        "params": {
            "side_a": {"type": "number", "default": None, "label": "Side a"},
            "side_b": {"type": "number", "default": None, "label": "Side b"},
            "side_c": {"type": "number", "default": None, "label": "Side c"},
            "angle_A": {"type": "number", "default": None, "label": "Angle A (deg)"},
            "angle_B": {"type": "number", "default": None, "label": "Angle B (deg)"},
            "angle_C": {"type": "number", "default": None, "label": "Angle C (deg)"},
        },
        "outputs": ["solved_sides", "solved_angles", "area", "perimeter"],
        "curriculum_topics": ["Law of sines", "Law of cosines", "Area of triangles using trig"],
    },
    "circle_theorems": {
        "name": "Circle Theorems Explorer",
        "category": "Geometry",
        "icon": "⭕",
        "description": "Tap a theorem, watch it animate with labeled arcs and chords",
        "params": {"theorem": {"type": "select", "options": [
            "inscribed_angle", "central_angle", "tangent_radius",
            "chord_chord", "secant_tangent", "thales"
        ]}},
        "outputs": ["animation_data", "theorem_statement", "proof_outline"],
        "curriculum_topics": ["Circle theorems", "Geometry"],
    },

    # ── Vectors & Linear Algebra ──────────────────────────────────────────
    "vector_canvas": {
        "name": "2D/3D Vector Canvas",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Add, scale, and dot-product vectors with labeled components",
        "params": {
            "vectors": {"type": "vector_list", "default": [[1, 0], [0, 1]]},
            "operation": {"type": "select", "options": ["add", "scale", "dot_product", "cross_product"]},
            "scalar": {"type": "number", "default": 1},
            "mode": {"type": "select", "options": ["2D", "3D"], "default": "2D"},
        },
        "outputs": ["result_vector", "magnitude", "angle", "visualization_data"],
        "curriculum_topics": ["Vector addition and scalar multiplication", "Dot product and cross product"],
    },
    "matrix_transform": {
        "name": "Matrix Transformation Grid",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Apply a 2×2 matrix to a point grid and see the geometric transformation (rotation, shear, reflection, scaling)",
        "params": {
            "matrix": {"type": "matrix", "rows": 2, "cols": 2, "default": [[1, 0], [0, 1]]},
            "grid_density": {"type": "slider", "min": 3, "max": 20, "default": 10},
        },
        "outputs": ["transformed_points", "determinant", "eigenvalues", "transform_type"],
        "curriculum_topics": ["Matrix as a linear transformation", "Rotation and reflection matrices"],
    },
    "eigenvalue_visualizer": {
        "name": "Eigenvalue Visualizer",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Show which directions a matrix stretches vs compresses",
        "params": {
            "matrix": {"type": "matrix", "rows": 2, "cols": 2, "default": [[2, 1], [1, 3]]},
        },
        "outputs": ["eigenvalues", "eigenvectors", "stretch_directions", "visualization_data"],
        "curriculum_topics": ["Eigenvalue equation", "Finding eigenvalues (characteristic polynomial)"],
    },
    "vector_span": {
        "name": "Vector Span & Basis",
        "category": "Vectors & Linear Algebra",
        "icon": "🔢",
        "description": "Interactive slider to build linear combinations and see the spanned space",
        "params": {
            "v1": {"type": "vector", "default": [1, 0]},
            "v2": {"type": "vector", "default": [0, 1]},
            "c1": {"type": "slider", "min": -3, "max": 3, "step": 0.1, "default": 1},
            "c2": {"type": "slider", "min": -3, "max": 3, "step": 0.1, "default": 1},
        },
        "outputs": ["result_vector", "spans_r2", "linearly_independent"],
        "curriculum_topics": ["Linear combinations and span", "Linear independence"],
    },

    # ── Calculus ──────────────────────────────────────────────────────────
    "derivative_animator": {
        "name": "Derivative Animator",
        "category": "Calculus",
        "icon": "∫",
        "description": "Sweep a tangent line along a curve with the slope value updating live",
        "params": {
            "expression": {"type": "text", "default": "x^2", "label": "f(x)"},
            "x_point": {"type": "slider", "min": -5, "max": 5, "step": 0.1, "default": 1},
            "x_range": {"type": "range", "default": [-5, 5]},
        },
        "outputs": ["curve_points", "tangent_line", "slope", "derivative_expression"],
        "curriculum_topics": ["Definition of the derivative", "Derivatives of trig functions"],
    },
    "riemann_sum": {
        "name": "Riemann Sum Builder",
        "category": "Calculus",
        "icon": "∫",
        "description": "Slider controls rectangle count, area estimate converges to integral",
        "params": {
            "expression": {"type": "text", "default": "x^2", "label": "f(x)"},
            "n_rectangles": {"type": "slider", "min": 1, "max": 200, "step": 1, "default": 10},
            "a": {"type": "number", "default": 0, "label": "Lower bound"},
            "b": {"type": "number", "default": 2, "label": "Upper bound"},
            "method": {"type": "select", "options": ["left", "right", "midpoint"], "default": "left"},
        },
        "outputs": ["rectangles", "area_estimate", "curve_points", "exact_integral"],
        "curriculum_topics": ["Definite integrals and area", "Fundamental theorem of calculus"],
    },
    "taylor_series": {
        "name": "Taylor Series Expansion",
        "category": "Calculus",
        "icon": "∫",
        "description": "Watch polynomials approximate sin/cos step by step",
        "params": {
            "function": {"type": "select", "options": ["sin", "cos", "exp", "log(1+x)", "1/(1-x)"], "default": "sin"},
            "n_terms": {"type": "slider", "min": 1, "max": 15, "step": 1, "default": 3},
            "center": {"type": "number", "default": 0, "label": "Expansion center a"},
            "x_range": {"type": "range", "default": [-6, 6]},
        },
        "outputs": ["original_curve", "approximation_curve", "polynomial_expression", "error_curve"],
        "curriculum_topics": ["Taylor and Maclaurin series", "Power series"],
    },
    "fourier_series": {
        "name": "Fourier Series",
        "category": "Calculus",
        "icon": "∫",
        "description": "Show how sine waves stack to form complex waveforms",
        "params": {
            "waveform": {"type": "select", "options": ["square", "sawtooth", "triangle", "custom"], "default": "square"},
            "n_harmonics": {"type": "slider", "min": 1, "max": 50, "step": 1, "default": 5},
        },
        "outputs": ["component_waves", "combined_wave", "coefficients"],
        "curriculum_topics": ["Fourier Series", "Sequences and convergence"],
    },
    "parametric_curves": {
        "name": "Parametric Curves",
        "category": "Calculus",
        "icon": "∫",
        "description": "Visualize curves defined by x(t) and y(t) as t sweeps",
        "params": {
            "x_expr": {"type": "text", "default": "cos(t)", "label": "x(t)"},
            "y_expr": {"type": "text", "default": "sin(t)", "label": "y(t)"},
            "t_range": {"type": "range", "default": [0, 6.28]},
            "t_current": {"type": "slider", "min": 0, "max": 6.28, "step": 0.05, "default": 3.14},
        },
        "outputs": ["curve_points", "current_point", "velocity_vector"],
        "curriculum_topics": ["Parametric curves"],
    },

    # ── Statistics ────────────────────────────────────────────────────────
    "bell_curve": {
        "name": "Normal Distribution (Bell Curve)",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Adjustable mean and standard deviation sliders, see the curve reshape live",
        "params": {
            "mu": {"type": "slider", "min": -5, "max": 5, "step": 0.1, "default": 0, "label": "Mean (mu)"},
            "sigma": {"type": "slider", "min": 0.1, "max": 3, "step": 0.1, "default": 1, "label": "Std Dev (sigma)"},
            "show_area": {"type": "toggle", "default": True, "label": "Show area under curve"},
        },
        "outputs": ["curve_points", "area_data", "percentiles"],
        "curriculum_topics": ["Normal distribution", "Standard deviation and variance"],
    },
    "regression_line": {
        "name": "Correlation & Regression",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Live scatter plot with a draggable regression line",
        "params": {
            "points": {"type": "point_list", "default": [[1, 2], [2, 4], [3, 5], [4, 4], [5, 5]]},
            "show_residuals": {"type": "toggle", "default": True},
        },
        "outputs": ["regression_coefficients", "r_squared", "residuals", "best_fit_line"],
        "curriculum_topics": ["Linear regression", "Correlation and scatter plots"],
    },
    "monte_carlo_pi": {
        "name": "Monte Carlo Pi Estimator",
        "category": "Statistics",
        "icon": "🎲",
        "description": "Random points, watch pi converge visually",
        "params": {
            "n_points": {"type": "slider", "min": 10, "max": 10000, "step": 10, "default": 1000},
            "speed": {"type": "slider", "min": 1, "max": 100, "step": 1, "default": 10},
        },
        "outputs": ["points_inside", "points_outside", "pi_estimate", "convergence_history"],
        "curriculum_topics": ["Monte Carlo simulation", "Basic probability rules"],
    },

    # ── 3D & Game Math ────────────────────────────────────────────────────
    "surface_3d": {
        "name": "3D Surface Plot",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "Render z = f(x,y) as a rotating mesh for terrain and physics",
        "params": {
            "expression": {"type": "text", "default": "sin(x)*cos(y)", "label": "z = f(x,y)"},
            "x_range": {"type": "range", "default": [-3, 3]},
            "y_range": {"type": "range", "default": [-3, 3]},
            "resolution": {"type": "slider", "min": 10, "max": 50, "default": 25},
        },
        "outputs": ["mesh_vertices", "mesh_faces", "gradient_colors"],
        "curriculum_topics": ["3D surface plots"],
    },
    "quaternion_rotation": {
        "name": "Quaternion Rotation Visualizer",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "How quaternions rotate objects in 3D without gimbal lock",
        "params": {
            "axis": {"type": "vector", "default": [0, 1, 0], "label": "Rotation axis"},
            "angle": {"type": "slider", "min": 0, "max": 360, "step": 1, "default": 45, "unit": "degrees"},
            "show_gimbal_comparison": {"type": "toggle", "default": True},
        },
        "outputs": ["quaternion", "rotation_matrix", "rotated_object", "euler_comparison"],
        "curriculum_topics": ["Quaternion rotation visualizer"],
    },
    "bezier_curves": {
        "name": "Bezier & Spline Curves",
        "category": "3D & Game Math",
        "icon": "🎮",
        "description": "Drag control points, watch smooth interpolated paths form",
        "params": {
            "control_points": {"type": "point_list", "default": [[0, 0], [1, 3], [3, 3], [4, 0]]},
            "curve_type": {"type": "select", "options": ["quadratic", "cubic", "catmull_rom"], "default": "cubic"},
            "t_resolution": {"type": "slider", "min": 10, "max": 200, "default": 100},
        },
        "outputs": ["curve_points", "tangent_at_t", "curvature"],
        "curriculum_topics": ["Bezier & spline curves"],
    },

    # ── Graph Theory & Discrete Math ──────────────────────────────────────
    "graph_traversal": {
        "name": "Graph Traversal (BFS/DFS)",
        "category": "Graph Theory",
        "icon": "🔗",
        "description": "Animate nodes being explored step-by-step",
        "params": {
            "algorithm": {"type": "select", "options": ["BFS", "DFS"], "default": "BFS"},
            "graph": {"type": "adjacency_list", "default": {"0": ["1", "2"], "1": ["3"], "2": ["3", "4"], "3": [], "4": []}},
            "start_node": {"type": "text", "default": "0"},
            "speed_ms": {"type": "slider", "min": 100, "max": 2000, "step": 100, "default": 500},
        },
        "outputs": ["traversal_order", "step_states", "visited_edges"],
        "curriculum_topics": ["BFS and DFS traversal", "Graph representations (adjacency list/matrix)"],
    },
    "fractal_explorer": {
        "name": "Fractal Explorer",
        "category": "Graph Theory",
        "icon": "🔗",
        "description": "Mandelbrot and Julia sets, procedurally generated",
        "params": {
            "fractal_type": {"type": "select", "options": ["mandelbrot", "julia"], "default": "mandelbrot"},
            "julia_c_real": {"type": "slider", "min": -2, "max": 2, "step": 0.01, "default": -0.7},
            "julia_c_imag": {"type": "slider", "min": -2, "max": 2, "step": 0.01, "default": 0.27015},
            "max_iterations": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "zoom": {"type": "slider", "min": 0.5, "max": 50, "step": 0.5, "default": 1},
        },
        "outputs": ["pixel_data", "iteration_counts"],
        "curriculum_topics": ["Fractals (Mandelbrot/Julia sets)"],
    },

    # ── Algebra ────────────────────────────────────────────────────────────
    "quadratic_explorer": {
        "name": "Quadratic Formula & Graph",
        "category": "Algebra",
        "icon": "📈",
        "description": "Enter a, b, c → see the parabola, vertex, axis of symmetry, roots via quadratic formula",
        "params": {
            "a": {"type": "slider", "min": -5, "max": 5, "step": 0.1, "default": 1, "label": "a"},
            "b": {"type": "slider", "min": -10, "max": 10, "step": 0.1, "default": 0, "label": "b"},
            "c": {"type": "slider", "min": -10, "max": 10, "step": 0.1, "default": -4, "label": "c"},
        },
        "outputs": ["roots", "vertex", "axis_of_symmetry", "discriminant", "parabola_points", "direction"],
        "curriculum_topics": ["Quadratic formula", "Completing the square", "Graphing quadratics", "Vertex form"],
    },

    # ── Scientific Calculator ─────────────────────────────────────────────
    "scientific_calculator": {
        "name": "Scientific Calculator",
        "category": "Tools",
        "icon": "🔬",
        "description": "Calculator with trig, log, summation (Σ), integrals (∫), imaginary numbers, e, π and more",
        "params": {
            "expression": {"type": "text", "default": "", "label": "Expression"},
        },
        "outputs": ["result", "latex", "steps", "visualization"],
        "curriculum_topics": [],
        "special_symbols": [
            {"symbol": "π", "insert": "pi", "label": "Pi"},
            {"symbol": "e", "insert": "E", "label": "Euler's number"},
            {"symbol": "i", "insert": "I", "label": "Imaginary unit"},
            {"symbol": "sin", "insert": "sin(", "label": "Sine"},
            {"symbol": "cos", "insert": "cos(", "label": "Cosine"},
            {"symbol": "tan", "insert": "tan(", "label": "Tangent"},
            {"symbol": "sin⁻¹", "insert": "asin(", "label": "Arcsine"},
            {"symbol": "cos⁻¹", "insert": "acos(", "label": "Arccosine"},
            {"symbol": "tan⁻¹", "insert": "atan(", "label": "Arctangent"},
            {"symbol": "ln", "insert": "ln(", "label": "Natural log"},
            {"symbol": "log", "insert": "log(", "label": "Log base 10"},
            {"symbol": "√", "insert": "sqrt(", "label": "Square root"},
            {"symbol": "|", "insert": "Abs(", "label": "Absolute value"},
            {"symbol": "x²", "insert": "**2", "label": "Square"},
            {"symbol": "xⁿ", "insert": "**", "label": "Power"},
            {"symbol": "n!", "insert": "factorial(", "label": "Factorial"},
            {"symbol": "Σ", "insert": "Sum(", "label": "Summation"},
            {"symbol": "∫", "insert": "integrate(", "label": "Integral"},
            {"symbol": "d/dx", "insert": "diff(", "label": "Derivative"},
            {"symbol": "∞", "insert": "oo", "label": "Infinity"},
        ],
    },
}


def get_visualization_modules() -> dict:
    """Return all available visualization modules grouped by category."""
    by_category = {}
    for mod_id, mod in VISUALIZATION_MODULES.items():
        cat = mod["category"]
        by_category.setdefault(cat, []).append({"id": mod_id} | mod)
    return by_category


def get_module(module_id: str) -> dict:
    """Return a single visualization module definition."""
    mod = VISUALIZATION_MODULES.get(module_id)
    if not mod:
        return {"error": f"Module '{module_id}' not found"}
    return {"id": module_id} | mod


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE BACKENDS — real math evaluation
# ═══════════════════════════════════════════════════════════════════════════════

import math
import cmath


def compute_quadratic(a: float, b: float, c: float) -> dict:
    """Compute quadratic formula results: roots, vertex, discriminant, parabola points."""
    if a == 0:
        # Linear: bx + c = 0
        if b == 0:
            return {"error": "Not a valid equation (a=0, b=0)", "roots": [], "vertex": None, "discriminant": None, "parabola_points": [], "direction": "none", "axis_of_symmetry": None}
        root = -c / b
        return {"roots": [root], "roots_display": [f"x = {root:.4g}"], "vertex": None, "discriminant": None, "axis_of_symmetry": None, "direction": "none",
                "parabola_points": [{"x": x / 5, "y": b * (x / 5) + c} for x in range(-50, 51)]}

    disc = b * b - 4 * a * c
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x * vertex_x + b * vertex_x + c

    roots = []
    roots_display = []
    if disc > 0:
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        roots = [r1, r2]
        roots_display = [f"x₁ = {r1:.4g}", f"x₂ = {r2:.4g}"]
    elif disc == 0:
        r = -b / (2 * a)
        roots = [r]
        roots_display = [f"x = {r:.4g} (repeated)"]
    else:
        # Complex roots
        real_part = -b / (2 * a)
        imag_part = math.sqrt(-disc) / (2 * a)
        roots_display = [f"x₁ = {real_part:.4g} + {imag_part:.4g}i", f"x₂ = {real_part:.4g} - {imag_part:.4g}i"]

    # Generate parabola points centered around vertex
    x_span = max(6, abs(vertex_x) + 5)
    x_min = vertex_x - x_span
    x_max = vertex_x + x_span
    step = (x_max - x_min) / 200
    parabola_points = []
    x = x_min
    while x <= x_max:
        y = a * x * x + b * x + c
        if abs(y) < 1e6:
            parabola_points.append({"x": round(x, 4), "y": round(y, 4)})
        x += step

    return {
        "roots": roots,
        "roots_display": roots_display,
        "vertex": {"x": round(vertex_x, 4), "y": round(vertex_y, 4)},
        "axis_of_symmetry": round(vertex_x, 4),
        "discriminant": round(disc, 4),
        "direction": "up" if a > 0 else "down",
        "parabola_points": parabola_points,
        "equation_display": f"f(x) = {a}x² + {b}x + {c}",
        "vertex_form": f"f(x) = {a}(x - {round(vertex_x, 4)})² + {round(vertex_y, 4)}",
        "formula_steps": [
            f"Discriminant: Δ = b² − 4ac = {b}² − 4({a})({c}) = {round(disc, 4)}",
            f"Vertex: x = −b/(2a) = −{b}/(2·{a}) = {round(vertex_x, 4)}",
            f"Vertex y: f({round(vertex_x, 4)}) = {round(vertex_y, 4)}",
        ] + ([f"Roots: x = (−b ± √Δ) / (2a) → {', '.join(roots_display)}"] if roots else [f"No real roots (Δ < 0): {', '.join(roots_display)}"]),
    }


def compute_scientific(expression: str) -> dict:
    """Evaluate a scientific expression using sympy. Returns result, LaTeX, and optional steps."""
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

    allowed_names = {
        "pi": sympy.pi, "e": sympy.E, "E": sympy.E, "I": sympy.I, "i": sympy.I,
        "oo": sympy.oo, "inf": sympy.oo,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
        "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
        "ln": sympy.ln, "log": sympy.log,
        "sqrt": sympy.sqrt, "Abs": sympy.Abs, "abs": sympy.Abs,
        "factorial": sympy.factorial,
        "Sum": sympy.Sum, "sum": sympy.Sum,
        "integrate": sympy.integrate, "diff": sympy.diff,
        "limit": sympy.limit, "solve": sympy.solve,
        "Product": sympy.Product, "product": sympy.Product,
        "x": sympy.Symbol("x"), "y": sympy.Symbol("y"),
        "n": sympy.Symbol("n"), "k": sympy.Symbol("k"),
        "t": sympy.Symbol("t"),
    }

    try:
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        parsed = parse_expr(expression, local_dict=allowed_names, transformations=transformations)

        # Evaluate unevaluated forms (Sum, Product) so they return a number
        if hasattr(parsed, 'doit'):
            parsed = parsed.doit()

        # Try to evaluate to a number
        result_exact = sympy.simplify(parsed)
        result_float = None
        try:
            f = complex(result_exact.evalf())
            if f.imag == 0:
                result_float = f.real
            else:
                result_float = f
        except (TypeError, ValueError, AttributeError):
            pass

        latex_str = sympy.latex(result_exact)
        input_latex = sympy.latex(parsed)

        return {
            "result": str(result_exact),
            "result_float": result_float if isinstance(result_float, (int, float)) else str(result_float) if result_float else None,
            "latex": latex_str,
            "input_latex": input_latex,
            "expression": expression,
        }
    except Exception as e:
        return {"error": str(e), "expression": expression}
