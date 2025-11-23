# README — Pipeline d’Analyse Automatisée des Tweets Free

**API Twitter → Nettoyage n8n → LLM Mistral Few-Shot Retriever → Réponse Auto → Chatbot → Ticketing**



## Introduction

Ce projet propose une solution complète et automatisée permettant de détecter, analyser et traiter les réclamations clients publiées sur Twitter/X à propos de Free.
L’objectif est d’assurer une prise en charge fluide, rapide et intelligente des problèmes exprimés sur les réseaux sociaux, tout en offrant une architecture robuste capable de fonctionner même en situation de crise.

La chaîne de traitement repose sur plusieurs briques clés : la collecte temps réel des tweets via l’API Twitter, un nettoyage avancé dans n8n, une classification basée sur un modèle de langage Mistral enrichi par un few-shot retriever, une réponse automatique aux clients, ainsi qu’un chatbot relié à une base de connaissances et un système interne de ticketing.

Une particularité centrale de la solution est son architecture hybride, permettant d’utiliser un LLM via API ou en local grâce à Ollama, garantissant une continuité opérationnelle même en cas de panne ou de surcharge.

## Architecture 
<img width="1198" height="639" alt="Diagramme sans nom (1)" src="https://github.com/user-attachments/assets/66a6fa49-4f2d-4453-9ba6-ff0a2874703a" />


## Collecte des Tweets via l’API Twitter / X

La première étape du pipeline consiste à capter automatiquement les tweets mentionnant Free. Pour cela, une requête programmée dans n8n interroge régulièrement l’API Twitter (v2) en utilisant des filtres tels que `@Free`, `@Freebox`, et différents hashtags liés aux problèmes réseau, fibre, box ou mobile.

L’API renvoie un ensemble d’informations structurées incluant le texte du tweet, son auteur, ses hashtags, son URL et d’autres métadonnées. Grâce à une fréquence d’interrogation d’environ 30 secondes, le système maintient une veille active et quasi temps réel des réclamations publiées par les utilisateurs.


## Nettoyage et Prétraitement des Tweets

Une fois collectés, les tweets bruts sont traités dans n8n pour assurer un nettoyage de haute qualité.
Cette étape est cruciale, car elle garantit que le texte analysé par le LLM est cohérent, sans bruit, et respecte les contraintes de confidentialité.

Le nettoyage inclut la suppression des URLs, le masquage des emails et numéros de téléphone, la normalisation des espaces et des sauts de ligne, la mise en minuscules et l’extraction rigoureuse des hashtags. Des règles de déduplication sont également appliquées afin d’éviter les doublons, en particulier lors de fortes affluences.

Ce processus produit un texte uniforme, standardisé et prêt pour l’analyse par le modèle de langage.

<img width="3572" height="164" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-11-21-141311" src="https://github.com/user-attachments/assets/01a8b16c-37bf-4f8a-ac08-4b3917041afe" />


## Classification avec LLM Mistral et Few-Shot Retriever

La classification des tweets est effectuée par un **modèle Mistral 7B**, déployé soit via API, soit en local avec **Ollama**.
Contrairement à un classifieur traditionnel, ce modèle agit comme un véritable interprète du langage naturel, capable de comprendre le contexte, l’ironie, les émotions et les signaux faibles propres aux interactions sur Twitter.

Pour optimiser la précision, un mécanisme de **few-shot retriever** sélectionne automatiquement les exemples les plus pertinents et les ajoute au prompt du modèle.
Cette stratégie permet au LLM de s’adapter dynamiquement au contenu du tweet, augmentant fortement la qualité des classifications, notamment pour les cas ambigus ou atypiques.

Le modèle renvoie ensuite un JSON structuré comprenant plusieurs champs tels que :

* la détection de réclamation,
* le type d’incident,
* le sentiment,
* l’urgence,
* les thématiques techniques (fibre, wifi, mobile…),
* un score de confiance.

Ce résultat permet de construire la logique de décision et de routage pour les étapes suivantes.


## Architecture Hybride : LLM via API + LLM Local

L’un des points forts de cette solution est sa capacité à fonctionner dans deux modes complémentaires :

1. **Mode normal : LLM via API**
   Dans des conditions standard, les requêtes sont envoyées au modèle Mistral hébergé dans le cloud.
   Ce mode offre rapidité, évolutivité et précision renforcée.

2. **Mode secours : LLM local avec Ollama**
   En cas de surcharge, de panne API, de quotas dépassés ou plus généralement lors d’un pic massif de tweets (panne majeure chez Free, incident réseau national…), le système bascule automatiquement sur un modèle local exécuté via Ollama.
   Ce modèle, basé sur Mistral 7B, permet de continuer à analyser les tweets **sans aucune dépendance extérieure**, assurant une continuité totale du service.

Le mécanisme de bascule est automatique : si l’API ne répond pas ou répond mal, le système active le mode local.
Cette approche garantit une résilience exceptionnelle et maintient l’activité du SAV même dans les situations critiques.

## Architecture N8N 
<img width="1037" height="388" alt="Capture d’écran (1803)" src="https://github.com/user-attachments/assets/b0038921-071a-4e45-83d7-aabca011cfd2" />

## Réponse Automatique aux Clients sur Twitter

Lorsqu’un tweet est identifié comme une réclamation, n8n génère une réponse automatique et la publie directement sous le tweet d’origine.
Cette réponse invite l’utilisateur à poursuivre la discussion avec un **chatbot d’assistance**, accessible via un lien inséré dans le commentaire.

Cette prise de contact immédiate améliore fortement la qualité du service client, permet une réduction des délais et assure une expérience fluide pour l’utilisateur.


## Chatbot Intelligent et Assistance Personnalisée

Le chatbot joue un rôle central dans la prise en charge du client.
Il commence par collecter trois informations obligatoires : le nom, le prénom et la description du problème.
Dès que ces informations sont transmises, un ticket interne est généré automatiquement.

Le chatbot utilise ensuite un modèle de langage associé à une base de connaissances (FAQ, documentation technique Free, procédures internes) pour tenter de résoudre le problème de manière autonome.
Si la solution est identifiée, le ticket est clos sans intervention humaine.

Si plusieurs tentatives échouent ou si le problème présente une urgence élevée, le ticket est escaladé automatiquement à un agent humain qui reprend la conversation avec l’historique complet.


## Création et Suivi des Tickets

Tous les tickets sont créés automatiquement suite aux interactions du chatbot. Ils regroupent :

* le tweet d’origine,
* les classifications du LLM,
* l’historique des échanges avec le client,
* le niveau d’urgence,
* le type d’incident,
* et l’état d’avancement (ouvert, en traitement, clos, escaladé).


Ce système garantit une gestion efficace, traçable et centralisée du support client Twitter/X.


## Interface Interne et KPIs

Une interface interne centralise l’ensemble des tickets et fournit des indicateurs clés pour piloter le SAV, tels que :
<img width="1316" height="516" alt="image" src="https://github.com/user-attachments/assets/43faca4f-1ab1-4f88-a39b-4df13c62cc50" />
* le taux de résolution automatique,
* le taux d’escalade,
* les délais moyens de résolution,
* l’évolution du volume de réclamations dans le temps,
* la répartition par type d’incident,
* l’analyse du sentiment des clients.

Ces KPIs permettent à l’entreprise d’améliorer en continu son service client et d’anticiper les problématiques récurrentes.


## Conclusion

Cette solution offre une automatisation complète, intelligente et résiliente de la gestion des réclamations Twitter liées à Free. L’utilisation combinée de l’API Twitter, d’un pipeline de nettoyage robuste, d’un modèle LLM avec few-shot retriever et d’une architecture hybride API + local garantit une performance optimale, y compris en période de crise.

Elle réduit significativement les délais de traitement, améliore la satisfaction client et donne une vision claire et précise de l’activité du SAV sur les réseaux sociaux.

