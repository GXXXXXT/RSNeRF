def get_kan_model(kan_basis_type='bspline'):
    if kan_basis_type == 'bspline':
        from .bspine_kan import BSpline_KAN
        return BSpline_KAN
    elif kan_basis_type == 'grbf':
        from .grbf_kan import GRBF_KAN
        return GRBF_KAN
    elif kan_basis_type == 'rbf':
        from .rbf_kan import RBF_KAN
        return RBF_KAN
    elif kan_basis_type == 'fourier':
        from .fourier_kan import Fourier_KAN
        return Fourier_KAN
    else:
        print("Not Implemented!!!")