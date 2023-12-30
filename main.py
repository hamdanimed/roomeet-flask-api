from flask import Flask,request,jsonify

from recommendation import get_recommendations
from recommendation import create_similarity_matrix

app = Flask(__name__)


# @app.route("/<user_id>",methods=['GET'])
# def getUser(user_id):
#     user_data={
#         "user_id":user_id,
#         "name":"hmad brahim"
#     }

#     extra = request.args.get("extra")
#     if extra:
#         user_data["extra"]=extra

#     return jsonify(user_data),200

# @app.route("/create-user",methods=['POST'])
# def create_user():
#     if request.method == 'POST':
#         data=request.get_json()

#     return jsonify(data),201

@app.route("/calculate-cosin-matrix",methods=['GET'])
def triggerCalculation():
    create_similarity_matrix()
    return jsonify("Done"),200

@app.route("/recommendation",methods=['POST'])
def recommendation():
    query=request.get_json()
    
    return jsonify(get_recommendations(query['Id'],query['nUsersToRecommend'])),200
    # return "\n".join(str(x) for x in import_data())


if __name__=="__main__":
    app.run(debug=False)