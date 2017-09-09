
# http://semver.org/
# I'll use "post" to indicate that the code currently follows (and
# may have broken things from) the pervious version.
__version__ = '0.1.0.post'

from VariationalBayes.Parameters import \
    ScalarParam, VectorParam, ArrayParam

from VariationalBayes.MatrixParameters import \
    PosDefMatrixParam, PosDefMatrixParamVector

from VariationalBayes.SimplexParams import SimplexParam

from VariationalBayes.ParameterDictionary import ModelParamsDict

from VariationalBayes.NormalParams import \
    UVNParam, UVNParamVector, MVNParam, MVNArray

from VariationalBayes.GammaParams import GammaParam

from VariationalBayes.WishartParams import WishartParam
