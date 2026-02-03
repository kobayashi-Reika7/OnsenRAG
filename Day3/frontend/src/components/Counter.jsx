/**
 * カウンター表示（未完了・完了・お気に入り・期限切れ）
 * 親から counts を受け取り表示するだけ（表示専用）
 */
import React from 'react';

function Counter({ counts }) {
  if (!counts) return null;
  return (
    <div className="counter-row">
      <span>未完了：{counts.incomplete}件</span>
      <span>完了：{counts.completed}件</span>
      <span>お気に入り：{counts.favorite}件</span>
      <span>期限切れ：{counts.overdue}件</span>
    </div>
  );
}

export default Counter;
