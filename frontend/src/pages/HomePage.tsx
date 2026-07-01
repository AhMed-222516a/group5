import { useNavigate } from "react-router-dom";
import { SparklesIcon, ArrowRightIcon } from "@heroicons/react/24/outline";

const features = [
  {
    title: "Semantic Retrieval",
    description:
      "FAISS-powered dense vector search finds the most relevant resolved tickets from the knowledge base.",
  },
  {
    title: "AI-Generated Responses",
    description:
      "Flan-T5 generates professional, context-grounded responses based on retrieved support history.",
  },
  {
    title: "Enterprise Ready",
    description:
      "Containerized with Docker, deployable to Azure, with a RESTful API for seamless integration.",
  },
];

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="max-w-4xl">
      <div className="bg-gradient-to-br from-blue-600 to-blue-800 rounded-2xl p-8 sm:p-12 text-white mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
            <SparklesIcon className="w-6 h-6" />
          </div>
          <span className="text-sm font-medium text-blue-100 uppercase tracking-wider">
            RAG System
          </span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold mb-3">
          Intelligent Support Ticket Classification
        </h1>
        <p className="text-blue-100 text-lg max-w-2xl mb-6 leading-relaxed">
          Automate customer support with AI-powered ticket classification and
          resolution. Our RAG system retrieves similar resolved tickets and
          generates accurate, context-aware responses.
        </p>
        <button
          onClick={() => navigate("/classify")}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-blue-700 font-semibold text-sm hover:bg-blue-50 transition-colors shadow-lg"
        >
          Start Classification
          <ArrowRightIcon className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {features.map((feature) => (
          <div
            key={feature.title}
            className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-5"
          >
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 text-sm mb-2">
              {feature.title}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
              {feature.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
