import { useState } from "react";
import {
  ClockIcon,
  BoltIcon,
  DocumentTextIcon,
  FolderIcon,
  ClipboardDocumentIcon,
  CheckIcon,
} from "@heroicons/react/24/outline";
import type { PredictionResponse, RetrievalResult } from "../types";

interface ResultCardProps {
  response: PredictionResponse;
}

function ConfidenceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? "bg-green-500" : pct >= 60 ? "bg-yellow-500" : "bg-orange-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 w-10 text-right">
        {pct}%
      </span>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
      <div className="flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
        <Icon className="w-4 h-4" />
        {label}
      </div>
      <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        {value}
      </p>
    </div>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };
  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
    >
      {copied ? (
        <CheckIcon className="w-3.5 h-3.5 text-green-500" />
      ) : (
        <ClipboardDocumentIcon className="w-3.5 h-3.5" />
      )}
      {copied ? "Copied" : `Copy ${label}`}
    </button>
  );
}

function SimilarTicketCard({ ticket }: { ticket: RetrievalResult }) {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
          Ticket #{ticket.ticket_id}
        </span>
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
          {Math.round(ticket.score * 100)}% match
        </span>
      </div>
      <div className="mb-2">
        <ConfidenceBar score={ticket.score} />
      </div>
      <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
        {ticket.subject}
      </h4>
      <div className="flex flex-wrap gap-2 mb-2">
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
          {ticket.category}
        </span>
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
          {ticket.priority}
        </span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
        {ticket.resolution_note}
      </p>
    </div>
  );
}

export default function ResultCard({ response }: ResultCardProps) {
  const allTicketsText = response.retrieval_results
    .map(
      (t) =>
        `[Ticket #${t.ticket_id}] ${t.subject} (${t.category}/${t.priority}) - ${t.resolution_note}`
    )
    .join("\n\n");

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <BoltIcon className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Classification Result
            </h2>
          </div>
          <div className="flex gap-1">
            <CopyButton text={response.answer} label="Response" />
            {allTicketsText && (
              <CopyButton text={allTicketsText} label="Tickets" />
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <StatCard
            icon={FolderIcon}
            label="Category"
            value={response.retrieval_results[0]?.category || "N/A"}
          />
          <StatCard
            icon={ClockIcon}
            label="Response Time"
            value={`${(response.total_latency * 1000).toFixed(0)}ms`}
          />
          <StatCard
            icon={DocumentTextIcon}
            label="Model"
            value={response.generation_model}
          />
          <StatCard
            icon={BoltIcon}
            label="Confidence"
            value={
              response.retrieval_results[0]
                ? `${Math.round(response.retrieval_results[0].score * 100)}%`
                : "N/A"
            }
          />
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Generated Response
          </h3>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-sm text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap">
            {response.answer}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Retrieved Similar Tickets
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {response.retrieval_results.map((ticket) => (
              <SimilarTicketCard key={ticket.rank} ticket={ticket} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
