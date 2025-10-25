import pickle

with open("db/year_dict.pkl", "rb") as f:
    year_dict = pickle.load(f)
    print("Year dictionary loaded successfully.")
    # year_dict example: '夏': '2209－1861 B.C.'

with open("db/place_dict.pkl", "rb") as f:
    place_dict = pickle.load(f)
    print("Place dictionary loaded successfully.")
    # place_dict example: '定軍山': '在陝西勉縣西南。諸葛亮葬此。'

with open("db/jiapu_dict.pkl", "rb") as f:
    jiapu_dict = pickle.load(f)
    print("Jiapu dictionary loaded successfully.")
    # jiapu_dict example: '家族': '與您有血緣關係的族人所形成的家庭組織。'

# Given a list of keywords, retrieve relevant information from the dictionaries


def retrieve_information(keywords):
    retrieved_info = []
    for keyword in keywords:
        if keyword in year_dict:
            retrieved_info.append(
                f"{keyword}: 年代是 {year_dict[keyword]}")
        if keyword in place_dict:
            retrieved_info.append(
                f"{keyword}: {place_dict[keyword]}")
        if keyword in jiapu_dict:
            retrieved_info.append(
                f"{keyword}: {jiapu_dict[keyword]}")
    return retrieved_info
