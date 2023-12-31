from flask import Flask,request,jsonify

from recommendation import get_recommendations
from recommendation import create_similarity_matrix_male
from recommendation import create_similarity_matrix_female

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/calculate-cosin-matrix-male",methods=['GET'])
def triggerCalculationMale():
    create_similarity_matrix_male()
    return jsonify("Done"),200

@app.route("/calculate-cosin-matrix-female",methods=['GET'])
def triggerCalculationFemale():
    create_similarity_matrix_female()
    return jsonify("Done"),200

@app.route("/recommendation",methods=['POST'])
def recommendation():
    query=request.get_json()
    
    recommendations=get_recommendations(query['userEmail'],query['nUsersToRecommend'])
    emails = [item[0] for item in recommendations]

    # Create a dictionary with 'usersEmails' as the key and the list of emails as the value
    result_dict = {'usersEmails': emails}
    return jsonify(result_dict),200
    # return "\n".join(str(x) for x in import_data())


if __name__=="__main__":
    app.run(debug=False)