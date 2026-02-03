/**
 * 1タスク用メモ（テキストエリア）
 * 入力変更を onBlur で親に渡し、親が API で保存する
 */
import React, { useState } from 'react';

function Memo({ value, onChange }) {
  const [local, setLocal] = useState(value || '');

  React.useEffect(() => {
    setLocal(value || '');
  }, [value]);

  const handleBlur = () => {
    if (onChange && local !== (value || '')) onChange(local);
  };

  return (
    <>
      <label className="memo-label">メモ：</label>
      <textarea
        className="memo-textarea"
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={handleBlur}
        placeholder="メモを入力"
      />
    </>
  );
}

export default Memo;
