function escapeCsv(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function rowsToCsv(rows) {
  if (!rows.length) return '';
  const fields = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return [
    fields.join(','),
    ...rows.map((row) => fields.map((field) => escapeCsv(row[field])).join(',')),
  ].join('\n');
}

export function downloadRows(filename, rows) {
  const csv = rowsToCsv(rows);
  if (!csv) return;
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
