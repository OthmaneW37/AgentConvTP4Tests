J'ai utilisé Claude pendant ce TP pour comprendre les problèmes où je suis resté bloqué.

Problèmes rencontrés :
- Git voulait convertir les fins de ligne des fichiers PDF. Claude m'a expliqué que
  cela pouvait les abîmer, puisqu'un PDF est un fichier binaire ; j'ai donc ajouté un
  fichier .gitattributes avec "*.pdf binary".
- Dans le test 4 (classe Test3), la question "Bonjour" utilisait quand même le RAG.
  Claude m'a aidé à comprendre que mon prompt était trop large : il demandait si la
  requête portait sur le RAG, le fine-tuning "ou l'IA", et le LM répondait "oui".
  J'ai limité le prompt au RAG et au fine-tuning.
- Pour le test 2, je ne comprenais pas pourquoi la réponse à "à quoi c'est comparé ?"
  était mauvaise alors que la mémoire était bien envoyée. Claude m'a expliqué que le
  ContentRetriever ne calcule l'embedding que de la dernière question, qui ne contient
  pas "fine-tuning".
- Pour le test 5, IntelliJ ne trouvait pas ma clé Tavily. Claude m'a expliqué
  qu'IntelliJ lit les variables d'environnement au démarrage ; après avoir mis la clé
  dans TAVILY_API_KEY, j'ai redémarré IntelliJ.
- Avec DefaultQueryRouter, des segments du cours étaient ajoutés à une question sur
  la Coupe d'Afrique des Nations. Claude m'a aidé à comprendre que ce routeur
  interroge toujours tous les retrievers ; je l'ai remplacé par un
  LanguageModelQueryRouter.
- Le modèle gemini-2.5-flash renvoyait une erreur. Claude m'a aidé à comprendre que
  ce modèle n'était plus disponible ; j'ai utilisé gemini-3.5-flash-lite.