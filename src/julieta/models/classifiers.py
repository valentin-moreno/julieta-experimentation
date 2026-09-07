"""Clasificadores propios del proyecto, reutilizables desde notebooks y pipelines.

Esta clase vive acá (y no dentro de un notebook) por una razón concreta y
no negociable: `joblib`/`pickle` no guardan el código de una clase, solo una
REFERENCIA del tipo `<modulo>.<NombreClase>`. Si la clase se define en una
celda de notebook, esa referencia queda como `__main__.BalancedXGBClassifier`,
que no existe en ningún otro proceso -- el .pkl resultante solo se puede
cargar desde ese mismo notebook y revienta con
`AttributeError: Can't get attribute 'BalancedXGBClassifier' on <module '__main__'>`
al intentar usarlo desde cualquier otro script.

Definida acá, la referencia guardada es
`julieta.models.classifiers.BalancedXGBClassifier`, que es importable desde
cualquier parte del proyecto.
"""

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier con `sample_weight` balanceado automático en cada `.fit()`.

    XGBoost, a diferencia de SVC/LogisticRegression/RandomForest, no tiene un
    parámetro `class_weight` nativo -- solo acepta `sample_weight` explícito en
    `.fit()`. Sobrescribiendo `.fit()` para calcular ese peso automáticamente a
    partir de la `y` que le llegue, este wrapper se comporta EXACTAMENTE igual
    que `XGBClassifier` en todo lo demás (mismos hiperparámetros, mismo
    `predict`/`predict_proba`).

    Además queda seguro para cross-validation: cada fold llama a `.fit()` con
    SOLO su porción de entrenamiento, así que el peso se calcula por fold y no
    hay fuga de información del fold de validación.

    `compute_sample_weight("balanced", y)` le da más peso a las filas de la
    clase minoritaria -- útil cuando una categoría de alta importancia clínica
    es también la menos representada en los datos.
    """

    def fit(self, X, y, **kwargs):
        sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)
