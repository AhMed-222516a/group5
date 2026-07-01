import { useState } from "react";
import { TrashIcon, MagnifyingGlassIcon, EyeIcon, XMarkIcon } from "@heroicons/react/24/outline";
import type { HistoryItem } from "../types";

interface HistoryTableProps {
  items: HistoryItem[];
  onDelete: (id: string) => void;
  onSearch: (query: string) => HistoryItem[];
}

function DetailModal({
  item,
  onClose,
}: {
  item: HistoryItem;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Prediction Details
          </h3>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800">
            <XMarkIcon className="w-5 h-5 text-gray-500" />
          </button>
        </div>
        <div className="space-y-4 text-sm">
          <div>
            <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Query</label>
            <p className="text-gray-900 dark:text-gray-100 bg-gray-50 dark:bg-gray-800 rounded-lg p-3">{item.request.query}</p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Response</label>
            <p className="text-gray-900 dark:text-gray-100 bg-gray-50 dark:bg-gray-800 rounded-lg p-3">{item.response.answer}</p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Category</label>
              <p className="text-gray-900 dark:text-gray-100">{item.response.retrieval_results[0]?.category || "N/A"}</p>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Latency</label>
              <p className="text-gray-900 dark:text-gray-100">{(item.response.total_latency * 1000).toFixed(0)}ms</p>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Model</label>
              <p className="text-gray-900 dark:text-gray-100">{item.response.generation_model}</p>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Date</label>
              <p className="text-gray-900 dark:text-gray-100">{new Date(item.timestamp).toLocaleString()}</p>
            </div>
          </div>
          {item.response.retrieval_results.length > 0 && (
            <div>
              <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Similar Tickets</label>
              <div className="space-y-2">
                {item.response.retrieval_results.map((t) => (
                  <div key={t.rank} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-xs">
                    <span className="font-medium text-blue-600">#{t.ticket_id}</span> - {t.subject} ({t.category})
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function HistoryTable({ items, onDelete, onSearch }: HistoryTableProps) {
  const [search, setSearch] = useState("");
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  const filtered = search ? onSearch(search) : items;

  return (
    <div>
      <div className="relative mb-4">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search predictions..."
          className="block w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
        />
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-sm text-gray-500">
          {search ? "No results match your search." : "No predictions yet."}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800">
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">Query</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider hidden sm:table-cell">Category</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider hidden md:table-cell">Latency</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider hidden lg:table-cell">Date</th>
                <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr
                  key={item.id}
                  className="border-b border-gray-100 dark:border-gray-800/50 hover:bg-gray-50 dark:hover:bg-gray-800/30 transition-colors"
                >
                  <td className="py-3 px-4 max-w-xs">
                    <p className="truncate text-gray-900 dark:text-gray-100 font-medium">
                      {item.request.query}
                    </p>
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                      {item.response.retrieval_results[0]?.category || "N/A"}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-500 dark:text-gray-400 hidden md:table-cell">
                    {(item.response.total_latency * 1000).toFixed(0)}ms
                  </td>
                  <td className="py-3 px-4 text-gray-500 dark:text-gray-400 hidden lg:table-cell text-xs">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => setSelectedItem(item)}
                        className="p-1.5 rounded-lg text-gray-400 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors"
                        title="View details"
                      >
                        <EyeIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDelete(item.id)}
                        className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                        title="Delete"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedItem && (
        <DetailModal item={selectedItem} onClose={() => setSelectedItem(null)} />
      )}
    </div>
  );
}
