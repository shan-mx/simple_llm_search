"use client";
import { Answer } from "@/app/components/answer";
import { Sources } from "@/app/components/sources";
import { Source } from "@/app/interfaces/source";
import { parseStreaming } from "@/app/utils/parse-streaming";
import { Annoyed } from "lucide-react";
import { FC, useEffect, useState, useRef } from "react";

export const Result: FC<{ query: string; rid: string }> = ({ query, rid }) => {
  const [sources, setSources] = useState<Source[]>([]);
  const [markdown, setMarkdown] = useState<string>("");
  const [error, setError] = useState<number | null>(null);
  const queryRef = useRef(true);

  useEffect(() => {
    if (queryRef.current) {
      queryRef.current = false;
      return;
    }
    const controller = new AbortController();
    void parseStreaming(
      controller,
      query,
      rid,
      setSources,
      setMarkdown,
      setError,
    );
    return () => {
      controller.abort();
    };
  }, [query]);
  return (
    <div className="flex flex-col gap-8">
      <Answer markdown={markdown} sources={sources}></Answer>
      <Sources sources={sources}></Sources>
      {error && (
        <div className="absolute inset-4 flex items-center justify-center bg-white/40 backdrop-blur-sm">
          <div className="p-4 bg-white shadow-2xl rounded text-blue-500 font-medium flex gap-4">
            <Annoyed></Annoyed>
            {error === 429
              ? "Sorry, you have made too many requests recently, try again later."
              : "Sorry, we might be overloaded, try again later."}
          </div>
        </div>
      )}
    </div>
  );
};
