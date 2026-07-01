import HistoryTable from "../components/HistoryTable";
import EmptyState from "../components/EmptyState";
import type { HistoryItem } from "../types";

interface HistoryPageProps {
  history: HistoryItem[];
  onDelete: (id: string) => void;
  onSearch: (query: string) => HistoryItem[];
  addToast: (message: string, type: "success" | "error" | "info") => void;
}

export default function HistoryPage({
  history,
  onDelete,
  onSearch,
  addToast,
}: HistoryPageProps) {
  const handleDelete = (id: string) => {
    onDelete(id);
    addToast("Prediction deleted", "info");
  };

  if (history.length === 0) {
    return (
      <div>
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            History
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            View your past predictions.
          </p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
          <EmptyState
            title="No prediction history"
            description="Your past predictions will appear here after you classify tickets."
          />
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          History
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          {history.length} prediction{history.length !== 1 ? "s" : ""} saved
        </p>
      </div>
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
        <HistoryTable
          items={history}
          onDelete={handleDelete}
          onSearch={onSearch}
        />
      </div>
    </div>
  );
}
