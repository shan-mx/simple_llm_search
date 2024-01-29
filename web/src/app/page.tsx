"use client";
import { Logo } from "@/app/components/logo";
import { PresetQuery } from "@/app/components/preset-query";
import { Search } from "@/app/components/search";
import React from "react";

export default function Home() {
  return (
    <div className="absolute inset-0 min-h-[500px] flex items-center justify-center">
      <div className="relative flex flex-col gap-8 px-4 -mt-24">
        <Logo></Logo>
        <Search></Search>
        <div className="flex gap-2 flex-wrap justify-center">
          <PresetQuery query="最近的新闻？"></PresetQuery>
          <PresetQuery query="OpenAI最新的模型"></PresetQuery>
          <PresetQuery query="原神有抄袭塞尔达吗"></PresetQuery>
        </div>
      </div>
    </div>
  );
}
