from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

#cnx mongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_default_database()

#modele d embedding 
print("Chargement du modèle d'embedding...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Modèle chargé !")

#chromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="offres",
    metadata={"hnsw:space": "cosine"}
)


def sync_offres():
    offres = list(db.offres.find())  #copie les offres de Mongo vers Chroma
    if not offres:
        print("Aucune offre trouvée dans MongoDB")
        return 0

    global collection
    try:
        chroma_client.delete_collection("offres") #supprime l'ancienne collection pour éviter les doublons
    except:
        pass
    collection = chroma_client.get_or_create_collection(
        name="offres",
        metadata={"hnsw:space": "cosine"}
    )

    ids, documents, metadatas = [], [], []
    for offre in offres: # ala kol offre, prend titre + domaine + description
        text = f"Poste: {offre.get('titre', '')}. Domaine: {offre.get('domaine', 'Autre')}. {offre.get('description', '')}"
        ids.append(str(offre["_id"]))
        documents.append(text)
        metadatas.append({
            "titre": offre.get("titre", ""),
            "domaine": offre.get("domaine", "Autre"),
            "description": offre.get("description", "")[:500]
        })

    embeddings = embed_model.encode(documents).tolist()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"{len(offres)} offres synchronisées dans ChromaDB")
    return len(offres)


def get_all_domaines():
    offres = list(db.offres.find())
    domaines = set()
    for o in offres:
        d = o.get("domaine", "Autre")
        if d:
            domaines.add(d)
    return list(domaines)


def build_response(question, results):
    q = question.lower()
    total = collection.count()

    #salut
    if any(mot in q for mot in ["bonjour", "salut", "hello", "bonsoir", "hi", "hey"]):
        domaines = get_all_domaines()
        return f"Bonjour ! Je suis l'assistant EasyRH.\n\nNous avons actuellement {total} offres disponibles dans les domaines : {', '.join(domaines)}.\n\nComment puis-je vous aider ?\n- Chercher des offres par domaine\n- Trouver un poste précis\n- Vous guider dans le processus de candidature"

    #merci 
    if any(mot in q for mot in ["merci", "thanks", "thank"]):
        return "De rien ! N'hésitez pas si vous avez d'autres questions. Bonne chance dans votre recherche d'emploi !"

    #temps de réponse 
    if any(mot in q for mot in ["combien de temps", "délai", "attendre", "temps de reponse", "temps de réponse"]):
        return "Délai de réponse :\n\nLe temps de réponse dépend du responsable RH de chaque entreprise. En général :\n\n- Les candidatures sont traitées sous quelques jours\n- Vous recevrez soit une acceptation (avec date d'entretien) soit un refus\n- Vous pouvez suivre le statut en temps réel dans 'Mes Candidatures'\n\nVous pouvez annuler une candidature en attente si vous changez d'avis."

    #processus de candidature
    if any(mot in q for mot in ["comment postuler", "comment candidater", "postuler", "candidater", "processus"]):
        return "Comment postuler sur EasyRH :\n\n1- Uploadez votre CV (format PDF) dans la section 'Mon CV'\n2- Consultez les offres disponibles dans 'Offres d'emploi'\n3- Cliquez sur 'Postuler' sur l'offre qui vous intéresse\n4- Votre CV sera automatiquement analysé par notre IA (score de compatibilité)\n5- Suivez le statut de votre candidature dans 'Mes Candidatures'\n\nLe RH pourra accepter votre candidature avec une date d'entretien ou la refuser avec un message explicatif."

    #CV 
    if any(mot in q for mot in ["cv", "curriculum", "upload", "déposer", "importer"]):
        return "Gestion de votre CV :\n\n- Allez dans la section 'Mon CV'\n- Cliquez ou glissez votre fichier PDF\n- Le texte sera automatiquement extrait pour le scoring IA\n- Vous pouvez remplacer votre CV à tout moment\n\nAstuce : Assurez-vous que votre CV est bien lisible (pas scanné en image) pour que le scoring IA fonctionne correctement !"

    #entretiens
    if any(mot in q for mot in ["entretien", "interview", "rendez-vous", "rdv"]):
        return "A propos des entretiens :\n\nQuand votre candidature est acceptée, le RH fixe une date d'entretien et vous envoie un message. Vous pouvez consulter ces informations dans 'Mes Candidatures'.\n\nConsultez régulièrement vos candidatures pour ne pas manquer une invitation !"

    #candidatures
    if any(mot in q for mot in ["candidature", "ma candidature", "mes candidatures"]):
        return "Délai de réponse :\n\nLe temps de réponse dépend du responsable RH. En général les candidatures sont traitées sous quelques jours.\n\nVous pouvez suivre le statut en temps réel dans 'Mes Candidatures' :\n- En attente : en cours d'examen\n- Acceptée : consultez la date d'entretien\n- Refusée : le RH a laissé un message\n\nVous pouvez annuler une candidature tant qu'elle est 'En attente'."

    #aide 
    if any(mot in q for mot in ["aide", "help", "quoi faire", "comment ça marche"]):
        return "Je peux vous aider à :\n\n- Chercher des offres : 'offres en informatique', 'poste de développeur'\n- Filtrer par domaine : 'offres en marketing', 'offres en finance'\n- Processus de candidature : 'comment postuler ?'\n- CV : 'comment uploader mon CV ?'\n- Questions générales : 'combien d'offres ?', 'quels domaines ?'\n\nPosez votre question !"

    #domaines 
    if any(mot in q for mot in ["domaine", "secteur", "catégorie", "categorie", "quels domaines"]):
        domaines = get_all_domaines()
        return f"Domaines disponibles sur EasyRH :\n\n{chr(10).join(['- ' + d for d in domaines])}\n\nTotal : {total} offres disponibles.\n\nVous pouvez me demander les offres d'un domaine précis, par exemple : 'offres en informatique'"

    #nombre d'offres 
    if any(mot in q for mot in ["combien", "nombre", "total", "count"]):
        domaines = get_all_domaines()
        return f"Nous avons actuellement {total} offres d'emploi disponibles sur EasyRH, réparties dans {len(domaines)} domaines : {', '.join(domaines)}.\n\nVoulez-vous que je cherche des offres dans un domaine précis ?"

    #score IA 
    if any(mot in q for mot in ["score", "scoring", "note", "compatibilité", "ia", "intelligence"]):
        return "Le Score IA (ATS) :\n\nQuand vous postulez à une offre, notre système d'intelligence artificielle analyse automatiquement votre CV et le compare avec la description de l'offre.\n\n- Score 70-100% : Excellent matching\n- Score 40-69% : Matching moyen\n- Score 0-39% : Faible matching\n\nLe score aide le RH à évaluer votre profil, mais ce n'est pas le seul critère de sélection !"

    #statuts de candidature 
    if any(mot in q for mot in ["statut", "status", "état", "en attente", "acceptée", "refusée"]):
        return "Les statuts de candidature :\n\n- En attente : Votre candidature est en cours d'examen\n- Acceptée : Félicitations ! Consultez la date d'entretien\n- Refusée : Le RH a laissé un message explicatif\n\nVous pouvez annuler une candidature tant qu'elle est 'En attente'."
    #recherche d'offres par mots-clés ou domaine 
    if not results["metadatas"] or not results["metadatas"][0]:
        return f"Désolé, je n'ai trouvé aucune offre correspondant à '{question}'.\n\nEssayez d'autres mots-clés ou tapez 'aide' !"

    offres = results["metadatas"][0]
    distances = results["distances"][0] if results["distances"] else []

    # Détecter si la question mentionne un domaine précis
    domaines_map = {
        "informatique": "Informatique",
        "it": "Informatique",
        "dev": "Informatique",
        "développeur": "Informatique",
        "developpeur": "Informatique",
        "programmeur": "Informatique",
        "web": "Informatique",
        "frontend": "Informatique",
        "backend": "Informatique",
        "fullstack": "Informatique",
        "full stack": "Informatique",
        "marketing": "Marketing",
        "commercial": "Commercial",
        "vente": "Commercial",
        "finance": "Finance",
        "comptable": "Finance",
        "comptabilité": "Finance",
        "rh": "RH",
        "ressources humaines": "RH",
        "design": "Design",
        "designer": "Design",
        "graphique": "Design",
        "juridique": "Juridique",
        "droit": "Juridique",
        "ingénierie": "Ingénierie",
        "ingenierie": "Ingénierie",
        "communication": "Communication",
    }

    domaine_filtre = None
    for mot, dom in domaines_map.items():
        if mot in q:
            domaine_filtre = dom
            break

    bonnes_offres = []
    for i, offre in enumerate(offres):
        dist = distances[i] if i < len(distances) else 1
        print(f"DEBUG -> Offre: {offre.get('titre')} | Domaine DB: '{offre.get('domaine')}' | Distance: {dist} | Filtre: {domaine_filtre}")

        if domaine_filtre:
            # Filtrer par domaine exact
            offre_domaine = offre.get("domaine", "").strip().lower()
            domaine_filtre_lower = domaine_filtre.strip().lower()
            if offre_domaine == domaine_filtre_lower or domaine_filtre_lower in offre_domaine or offre_domaine in domaine_filtre_lower:
                bonnes_offres.append(offre)
        else:
            # Pas de domaine mentionné -> filtrer par distance sémantique
            if dist < 0.6:
                bonnes_offres.append(offre)

    if not bonnes_offres:
        if domaine_filtre:
            return f"Désolé, aucune offre n'est disponible dans le domaine '{domaine_filtre}' pour le moment.\n\nNous avons {total} offres dans d'autres domaines. Tapez 'domaines' pour voir la liste !"
        return f"Désolé, je n'ai trouvé aucune offre correspondant à '{question}'.\n\nEssayez avec d'autres mots-clés !"

    response = f"J'ai trouvé {len(bonnes_offres)} offre(s) correspondant à votre recherche :\n\n"
    for i, offre in enumerate(bonnes_offres):
        response += f"- {offre['titre']}\n"
        response += f"  Domaine : {offre['domaine']}\n"
        desc = offre['description'][:150]
        if len(offre['description']) > 150:
            desc += "..."
        response += f"  {desc}\n\n"

    response += "Pour postuler, rendez-vous dans 'Offres d'emploi' et cliquez sur 'Postuler' !"
    return response


# ROUTES 

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"success": False, "message": "La question est vide"}), 400

    query_embedding = embed_model.encode([question]).tolist()

    count = collection.count()
    if count == 0:
        return jsonify({
            "success": True,
            "response": "Aucune offre n'est disponible pour le moment. Revenez bientôt !",
            "sources": []
        })

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(5, count)
    )

    response = build_response(question, results)

    sources = []
    if results["metadatas"] and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            sources.append({"titre": meta["titre"], "domaine": meta["domaine"]})

    return jsonify({
        "success": True,
        "response": response,
        "sources": sources
    })


@app.route("/api/sync", methods=["POST"])
def sync():
    count = sync_offres()
    return jsonify({"success": True, "message": f"{count} offres synchronisées"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "offres_indexees": collection.count()})


if __name__ == "__main__":
    print("Synchronisation des offres...")
    sync_offres()
    print("Chatbot RAG démarré sur http://localhost:5001")
    app.run(port=5001, debug=False) 
    