class IdList:

    def __init__(self, traces):
        self.ids = []       # Preliminary Phase IDs
        self.tids = []      # Trace IDs
        self.idList = {}    # Actual Data Structure
                            #           ID
                            # tid   |   Occurence
        self.RXMap = {}     # Right-eXtention-Map indicate pairwise extention relationship
        self.LXMap = {}     # Left-eXtention-Map indicate pairwise extention relationship
        self.max_gap = 1        # Gap is 1 by default

        self.build_list(self, traces)

    # Construct the Id List given the list of preliminary phases as list of sets: [{phase_1}, {phase_2}, ..., {phase_n}]
    def build_list(self, traces):
        if not isinstance(traces, list):
            print('Input Traces must be list object.')
            raise TypeError

        # Itterate Each Trace
        trace_idx = 0
        for trace in traces:
            for event_idx in range(len(trace)):
                phase = trace[event_idx]

                # Check if phase already exists
                if phase not in ids:
                    idList[len(ids)] = [[] for _ in range(len(traces))]
                    RXMap[len(ids)] = set()
                    LXMap[len(ids)] = set()
                    ids.append(phase)

                phase_idx = ids.index(phase)

                # Insert to IdList
                idList[phase_idx][trace_idx].append(component_idx)

                # Update Extention Map
                for gap in range(1, max_gap + 1):
                    # Left Extention
                    idx = component_idx - gap
                    if idx >= 0:
                        LXMap[phase_idx].add(ids.index(components[idx]))
                    # Righ Extention
                    idx = component_idx + gap
                    if idx < len(components):
                        # Check if phase already exists in ids
                        if components[idx] not in ids:
                            idList[len(ids)] = [[] for _ in range(len(traces))]
                            LXMap[len(ids)] = set()
                            RXMap[len(ids)] = set()
                            ids.append(components[idx])
                        RXMap[phase_idx].add(ids.index(components[idx]))

