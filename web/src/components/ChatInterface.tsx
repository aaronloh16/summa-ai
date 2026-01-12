"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";

// Types matching the API response
interface SearchResult {
  chunk_id: string;
  work: string;
  work_abbrev: string;
  location: string;
  title: string;
  text: string;
  score: number;
  source_url?: string;
}

interface ChatResponse {
  response: string;
  sources: SearchResult[];
}

interface Work {
  abbrev: string;
  name: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SearchResult[];
  isLoading?: boolean;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Work abbreviation colors
const workColors: Record<string, string> = {
  ST: "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-700",
  SCG: "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-700",
  DV: "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  DPD: "bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-700",
  QDA: "bg-rose-100 text-rose-800 border-rose-200 dark:bg-rose-900/30 dark:text-rose-300 dark:border-rose-700",
  DBE: "bg-cyan-100 text-cyan-800 border-cyan-200 dark:bg-cyan-900/30 dark:text-cyan-300 dark:border-cyan-700",
  CM: "bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-700",
};

const WORKS: Work[] = [
  { abbrev: "ST", name: "Summa Theologica" },
  { abbrev: "SCG", name: "Summa Contra Gentiles" },
  { abbrev: "DV", name: "De Veritate" },
  { abbrev: "DPD", name: "De Potentia Dei" },
  { abbrev: "QDA", name: "Quaestiones de Anima" },
  { abbrev: "DBE", name: "On Being and Essence" },
  { abbrev: "CM", name: "Commentary on Metaphysics" },
];

// Aquinas quotes for the thinking animation
const THINKING_QUOTES = [
  "Searching the sacred texts...",
  "Consulting the Summa...",
  "Distinguo...",
  "Pondering the objections...",
  "Reviewing the sed contra...",
  "Formulating the respondeo...",
  "Examining the authorities...",
  "Considering the matter carefully...",
];

function SourceCard({ source, index }: { source: SearchResult; index: number }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-lg border bg-[var(--card)] overflow-hidden transition-all duration-200 hover:border-[var(--accent)]/30 hover:shadow-sm">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-[var(--muted)]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/20">
            {index + 1}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span
                className={`inline-block px-1.5 py-0.5 text-[10px] uppercase tracking-wider font-semibold rounded border ${
                  workColors[source.work_abbrev] || "bg-gray-100 text-gray-800 border-gray-200"
                }`}
              >
                {source.work_abbrev}
              </span>
              <p className="text-sm font-medium truncate">{source.location}</p>
            </div>
            <p className="text-xs text-[var(--muted-foreground)] mt-0.5 truncate max-w-xs">
              {source.title}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-[var(--muted-foreground)] tabular-nums font-medium">
            {(source.score * 100).toFixed(0)}%
          </span>
          <svg
            className={`w-4 h-4 text-[var(--muted-foreground)] transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>
      {isExpanded && (
        <div className="px-4 py-3 border-t bg-[var(--muted)]/30 max-h-72 overflow-y-auto">
          <p className="text-sm whitespace-pre-wrap leading-relaxed text-[var(--card-foreground)]/80 font-serif">
            {source.text}
          </p>
          {source.source_url && (
            <a
              href={source.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 mt-4 px-3 py-1.5 text-xs text-[var(--accent)] hover:bg-[var(--accent)]/10 rounded-md transition-colors"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              View original source
            </a>
          )}
        </div>
      )}
    </div>
  );
}

function ThinkingAnimation() {
  const [quoteIndex, setQuoteIndex] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setQuoteIndex((prev) => (prev + 1) % THINKING_QUOTES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-start gap-4 py-4">
      {/* Animated seal */}
      <div className="relative flex-shrink-0">
        <div className="w-10 h-10 rounded-full bg-[var(--muted)] flex items-center justify-center overflow-hidden">
          <Image 
            src="/Seal_of_the_Dominican_Order.svg" 
            alt="Thinking" 
            width={32} 
            height={32}
            className="opacity-70 animate-pulse"
          />
        </div>
        {/* Rotating ring */}
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-[var(--accent)] animate-spin" style={{ animationDuration: "1.5s" }} />
      </div>
      
      <div className="flex-1 pt-2">
        {/* Thinking text with fade transition */}
        <p 
          key={quoteIndex}
          className="text-sm italic text-[var(--muted-foreground)] animate-fade-in"
        >
          {THINKING_QUOTES[quoteIndex]}
        </p>
        
        {/* Animated dots */}
        <div className="flex gap-1.5 mt-3">
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]/60 animate-bounce" style={{ animationDelay: "0ms", animationDuration: "1s" }} />
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]/60 animate-bounce" style={{ animationDelay: "150ms", animationDuration: "1s" }} />
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]/60 animate-bounce" style={{ animationDelay: "300ms", animationDuration: "1s" }} />
        </div>
      </div>
    </div>
  );
}

function WorkFilter({ 
  selectedWork, 
  onSelect 
}: { 
  selectedWork: string | null; 
  onSelect: (work: string | null) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border bg-[var(--card)] hover:bg-[var(--muted)]/50 transition-colors"
      >
        <svg className="w-3.5 h-3.5 text-[var(--muted-foreground)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
        </svg>
        <span className="font-medium">{selectedWork ? WORKS.find(w => w.abbrev === selectedWork)?.name : "All Works"}</span>
        <svg className={`w-3 h-3 transition-transform ${isOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full right-0 mt-1 w-64 bg-[var(--card)] rounded-lg border shadow-xl z-20 py-1 animate-fade-in">
            <button
              onClick={() => { onSelect(null); setIsOpen(false); }}
              className={`w-full px-3 py-2.5 text-left text-sm hover:bg-[var(--muted)]/50 transition-colors ${
                !selectedWork ? "bg-[var(--muted)]/30 font-medium" : ""
              }`}
            >
              All Works
            </button>
            <div className="h-px bg-[var(--border)] my-1" />
            {WORKS.map(work => (
              <button
                key={work.abbrev}
                onClick={() => { onSelect(work.abbrev); setIsOpen(false); }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-[var(--muted)]/50 transition-colors flex items-center gap-2.5 ${
                  selectedWork === work.abbrev ? "bg-[var(--muted)]/30 font-medium" : ""
                }`}
              >
                <span className={`inline-block w-10 px-1.5 py-0.5 text-[9px] text-center uppercase tracking-wider font-semibold rounded border ${
                  workColors[work.abbrev] || "bg-gray-100 text-gray-800 border-gray-200"
                }`}>
                  {work.abbrev}
                </span>
                <span>{work.name}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// Markdown renderer component with custom styling
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => (
          <p className="mb-4 last:mb-0 leading-relaxed">{children}</p>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-[var(--card-foreground)]">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="italic">{children}</em>
        ),
        h1: ({ children }) => (
          <h1 className="text-xl font-bold mb-3 mt-4 first:mt-0">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-lg font-bold mb-2 mt-4 first:mt-0">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h3>
        ),
        ul: ({ children }) => (
          <ul className="list-disc pl-5 mb-4 space-y-1">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-5 mb-4 space-y-1">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">{children}</li>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-3 border-[var(--accent)] pl-4 py-1 my-4 italic bg-[var(--muted)]/30 rounded-r-lg">
            {children}
          </blockquote>
        ),
        code: ({ children }) => (
          <code className="px-1.5 py-0.5 bg-[var(--muted)] rounded text-sm font-mono">{children}</code>
        ),
        a: ({ href, children }) => (
          <a href={href} className="text-[var(--accent)] hover:underline" target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [workFilter, setWorkFilter] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    // Add loading message
    const loadingId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: loadingId, role: "assistant", content: "", isLoading: true },
    ]);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: userMessage.content, 
          top_k: 5,
          work_filter: workFilter
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data: ChatResponse = await res.json();

      // Replace loading message with actual response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingId
            ? {
                id: loadingId,
                role: "assistant",
                content: data.response,
                sources: data.sources,
                isLoading: false,
              }
            : msg
        )
      );
    } catch (err) {
      // Remove loading message and show error
      setMessages((prev) => prev.filter((msg) => msg.id !== loadingId));
      setError(err instanceof Error ? err.message : "Failed to get response");
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  const suggestedQuestions = [
    "What is the nature of truth?",
    "What are the five ways to prove God exists?",
    "What is the essence of happiness?",
    "How does Aquinas explain divine power?",
  ];

  return (
    <div className="flex flex-col h-screen max-h-screen bg-[var(--background)] grain">
      {/* Header */}
      <header className="flex-shrink-0 border-b bg-[var(--card)]/90 backdrop-blur-md px-6 py-4 shadow-sm">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-full overflow-hidden bg-[var(--muted)] flex items-center justify-center p-1.5 shadow-inner">
              <Image 
                src="/Seal_of_the_Dominican_Order.svg" 
                alt="Dominican Order Seal" 
                width={40} 
                height={40}
                className="opacity-90"
              />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Aquinas RAG</h1>
              <p className="text-xs text-[var(--muted-foreground)] tracking-wide">The Angelic Doctor</p>
            </div>
          </div>
          <WorkFilter selectedWork={workFilter} onSelect={setWorkFilter} />
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-3xl mx-auto space-y-8">
          {messages.length === 0 && (
            <div className="text-center py-16 animate-slide-up">
              <div className="w-28 h-28 mx-auto mb-8 rounded-full bg-gradient-to-br from-[var(--muted)] to-[var(--muted)]/50 flex items-center justify-center p-4 shadow-lg">
                <Image 
                  src="/Seal_of_the_Dominican_Order.svg" 
                  alt="Dominican Order Seal" 
                  width={80} 
                  height={80}
                  className="opacity-85"
                />
              </div>
              <h2 className="text-2xl font-bold mb-3 tracking-tight">Ask the Angelic Doctor</h2>
              <p className="text-[var(--muted-foreground)] max-w-md mx-auto mb-4 leading-relaxed">
                Explore the theological and philosophical wisdom of Thomas Aquinas through conversation.
              </p>
              <p className="text-xs text-[var(--muted-foreground)]/70 mb-12 max-w-sm mx-auto">
                Search across 11,000+ passages from the Summa Theologica, Summa Contra Gentiles, De Veritate, and more.
              </p>
              <div className="flex flex-col sm:flex-row flex-wrap justify-center gap-2.5">
                {suggestedQuestions.map((q) => (
                  <button
                    key={q}
                    onClick={() => setInput(q)}
                    className="px-5 py-2.5 text-sm rounded-full border bg-[var(--card)] hover:bg-[var(--muted)] hover:border-[var(--accent)]/40 hover:shadow-md transition-all duration-200"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, idx) => (
            <div
              key={message.id}
              className={`animate-fade-in ${message.role === "user" ? "flex justify-end" : ""}`}
              style={{ animationDelay: `${idx * 50}ms` }}
            >
              {message.role === "user" ? (
                <div className="max-w-[85%] px-5 py-3 rounded-2xl rounded-br-sm bg-[var(--accent)] text-[var(--accent-foreground)] shadow-md">
                  <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                </div>
              ) : (
                <div className="space-y-5">
                  {message.isLoading ? (
                    <ThinkingAnimation />
                  ) : (
                    <>
                      <div className="prose prose-stone dark:prose-invert prose-sm max-w-none text-[var(--card-foreground)]">
                        <MarkdownContent content={message.content} />
                      </div>
                      {message.sources && message.sources.length > 0 && (
                        <div className="space-y-2.5 pt-3">
                          <h4 className="text-[11px] font-bold text-[var(--muted-foreground)] uppercase tracking-widest flex items-center gap-2">
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                            </svg>
                            Sources ({message.sources.length})
                          </h4>
                          <div className="grid gap-2">
                            {message.sources.map((source, idx) => (
                              <SourceCard key={`${source.chunk_id}-${idx}`} source={source} index={idx} />
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          ))}

          {error && (
            <div className="animate-fade-in p-4 rounded-xl bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800/30 text-red-700 dark:text-red-400 shadow-sm">
              <div className="flex items-center gap-2 mb-1">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="font-semibold text-sm">Connection Error</p>
              </div>
              <p className="text-sm opacity-80">{error}</p>
              <p className="text-xs mt-2 opacity-60">Make sure the API server is running at {API_BASE}</p>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t bg-[var(--card)]/90 backdrop-blur-md px-6 py-4 shadow-[0_-2px_10px_rgba(0,0,0,0.05)]">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={workFilter 
                  ? `Ask about ${WORKS.find(w => w.abbrev === workFilter)?.name}...`
                  : "Ask about Aquinas's works..."
                }
                rows={1}
                disabled={isLoading}
                className="w-full px-4 py-3 rounded-xl border bg-[var(--background)] resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent)] disabled:opacity-50 transition-all text-sm shadow-inner"
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 w-11 h-11 rounded-xl bg-[var(--accent)] text-[var(--accent-foreground)] flex items-center justify-center transition-all hover:brightness-110 hover:shadow-md disabled:opacity-40 disabled:cursor-not-allowed shadow-sm active:scale-95"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
          <p className="mt-2.5 text-[10px] text-center text-[var(--muted-foreground)]/70 tracking-wide">
            Press Enter to send · Shift+Enter for new line
          </p>
        </form>
      </footer>
    </div>
  );
}
