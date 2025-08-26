import React, { useState, useEffect } from 'react';
import { Domain } from '@/types/data';

interface DomainFilterProps {
  domains: Domain[];
  onDomainsChange: (domains: Domain[]) => void;
  isVisible: boolean;
}

/**
 * DomainFilter component for managing domain filtering in research tasks.
 * 
 * This component allows users to add and remove domain filters that restrict
 * research to specific websites or sources. It provides a clean interface
 * for domain management with local storage persistence.
 * 
 * @param domains - Array of current domain filters
 * @param onDomainsChange - Callback function called when domains change
 * @param isVisible - Whether the component should be displayed
 */
export default function DomainFilter({ 
  domains, 
  onDomainsChange, 
  isVisible 
}: DomainFilterProps) {
  const [newDomain, setNewDomain] = useState<string>('');

  // Load domains from localStorage on component mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('domainFilters');
      if (saved) {
        try {
          const parsedDomains = JSON.parse(saved);
          if (Array.isArray(parsedDomains) && parsedDomains.length > 0) {
            onDomainsChange(parsedDomains);
          }
        } catch (error) {
          console.warn('Failed to parse saved domain filters:', error);
          localStorage.removeItem('domainFilters');
        }
      }
    }
  }, [onDomainsChange]);

  // Save domains to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('domainFilters', JSON.stringify(domains));
    }
  }, [domains]);

  /**
   * Handles adding a new domain to the filter list.
   * Validates the domain input and prevents duplicates.
   */
  const handleAddDomain = (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedDomain = newDomain.trim();
    if (!trimmedDomain) {
      return;
    }

    // Check for duplicates
    const domainExists = domains.some(
      domain => domain.value.toLowerCase() === trimmedDomain.toLowerCase()
    );

    if (domainExists) {
      console.warn('Domain already exists in filter list');
      return;
    }

    // Add the new domain
    const updatedDomains = [...domains, { value: trimmedDomain }];
    onDomainsChange(updatedDomains);
    setNewDomain('');
  };

  /**
   * Handles removing a domain from the filter list.
   * 
   * @param domainToRemove - The domain value to remove
   */
  const handleRemoveDomain = (domainToRemove: string) => {
    const updatedDomains = domains.filter(
      domain => domain.value !== domainToRemove
    );
    onDomainsChange(updatedDomains);
  };

  /**
   * Handles keyboard events in the domain input field.
   * Submits the form when Enter is pressed.
   */
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddDomain(e as any);
    }
  };

  // Don't render if not visible
  if (!isVisible) {
    return null;
  }

  return (
    <div className="mt-4 domain_filters">
      <div className="flex gap-2 mb-4">
        <label htmlFor="domain_filters" className="agent_question">
          Filter by domain{" "}
        </label>
        <input
          type="text"
          value={newDomain}
          onChange={(e) => setNewDomain(e.target.value)}
          placeholder="Filter by domain (e.g., techcrunch.com)"
          className="input-static"
          onKeyPress={handleKeyPress}
          aria-label="Add domain filter"
        />
        <button
          type="button"
          onClick={handleAddDomain}
          className="button-static"
          disabled={!newDomain.trim()}
          aria-label="Add domain filter"
        >
          Add Domain
        </button>
      </div>

      {domains.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {domains.map((domain, index) => (
            <div
              key={`${domain.value}-${index}`}
              className="domain-tag-static"
            >
              <span className="domain-text-static">{domain.value}</span>
              <button
                type="button"
                onClick={() => handleRemoveDomain(domain.value)}
                className="domain-button-static"
                aria-label={`Remove ${domain.value} filter`}
                title={`Remove ${domain.value} filter`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {domains.length === 0 && (
        <p className="text-sm text-gray-500 mt-2">
          No domain filters active. Add domains to restrict research to specific sources.
        </p>
      )}
    </div>
  );
}