export default function MdText({ text }) {
  if (!text) return null;

  const renderInline = (segment, keyPrefix) => {
    const parts = segment.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return parts.map((part, index) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={`${keyPrefix}-strong-${index}`}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={`${keyPrefix}-code-${index}`}>{part.slice(1, -1)}</code>;
      }
      return <span key={`${keyPrefix}-text-${index}`}>{part}</span>;
    });
  };

  const blocks = text.split(/\n\s*\n/).map(block => block.trim()).filter(Boolean);

  return (
    <div className="md-text">
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n").map(line => line.trim()).filter(Boolean);
        const isList = lines.length > 0 && lines.every(line => /^(- |\* |\d+\. )/.test(line));

        if (isList) {
          return (
            <ul key={`block-${blockIndex}`}>
              {lines.map((line, lineIndex) => (
                <li key={`block-${blockIndex}-item-${lineIndex}`}>
                  {renderInline(line.replace(/^(- |\* |\d+\. )/, ""), `block-${blockIndex}-item-${lineIndex}`)}
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={`block-${blockIndex}`}>
            {lines.map((line, lineIndex) => (
              <span key={`block-${blockIndex}-line-${lineIndex}`}>
                {renderInline(line, `block-${blockIndex}-line-${lineIndex}`)}
                {lineIndex < lines.length - 1 && <br />}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}
