import json

from marlena.marlena import MARLENA
import pandas as pd
import numpy as np
from sklearn.externals import joblib


from aiohttp import web
from aiohttp_swagger import *


async def marlena(request):
    """
        ---
        description: Marlena
        tags:
            - marlena
        produces:
            - application/json
        responses:
            "200":
                description: successful operation.
            "500":
                description: operation failed


        """
    print("here", request)
    try:
        bb = joblib.load('./black_boxes/RandomForest.pkl')
        numerical_vars = np.load('./data/numerical_vars.npy').tolist()
        labels_name = np.load('./data/labels_name.npy').tolist()
        categorical_vars = np.load('./data/categorical_vars.npy').tolist()
        df = pd.read_csv('./data/preprocessed_patient-characteristics.csv')
        m1 = MARLENA(neigh_type='unified')
        i2e = df.loc[3, numerical_vars + categorical_vars]
        rule, instance_imporant_feat, fidelity, hit, DT = m1.extract_explanation(i2e, df, bb, numerical_vars,
                                                                                 categorical_vars, labels_name, k=200,
                                                                                 size=2000, alpha=0.1)

        res = {"rule": rule, 'feat_importance': instance_imporant_feat, 'fidelity': fidelity, 'hit': hit}

        print(res)

        return web.Response(text=json.dumps(res), status=200)

    except Exception as e:
        response_obj = dict(status='failure', description=str(e))
        return web.Response(text=json.dumps(response_obj), status=500)


async def make_app():
    app = web.Application()

    # add the routes
    app.add_routes([
        web.post('/api/marlena', marlena),
    ])

    setup_swagger(app, swagger_url="/api/v1/doc", description="",
                  title="Marlena Server API",
                  api_version="0.0.1",
                  contact="giulio.rossetti@gmail.com")

    return app


if __name__ == '__main__':
    web.run_app(make_app(), port=8081, host="0.0.0.0")
