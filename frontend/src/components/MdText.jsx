export default function MdText({ text }) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <p>
      {parts.map((p, i) => {
        if (p.startsWith("**") && p.endsWith("**")) return <strong key={i}>{p.slice(2, -2)}</strong>;
        if (p.startsWith("`") && p.endsWith("`")) return <code key={i}>{p.slice(1, -1)}</code>;
        const lines = p.split("\n");
        return lines.map((line, j) => (
          <span key={`${i}-${j}`}>{line}{j < lines.length - 1 && <br />}</span>
        ));
      })}
    </p>
  );
}
