import face_recognition
import cv2
import pickle

class FaceDetectorMemory(object):

    def __init__(self, picklePath=None):
        if picklePath is None:
            # initalize empty memory
            self.knownFaceEmbeddings = list()
            self.knownNames = list()
        else:
            self.load(picklePath)
    

    def save(self, picklePath):
        tmp = {"knownNames": self.knownNames, "knownFaces": self.knownFaceEmbeddings}
        with open(picklePath, "w") as handle:
            pickle.dump(tmp, handle)

    def load(self, picklePath):
        with open(picklePath, "r") as handle:
            tmp = pickle.load(handle)
            self.knownFaceEmbeddings = tmp["knownFaces"]
            self.knownNames = tmp["knownNames"]

    def _countDuplicates(self, newName):
        dups = 0
        for name in self.knownNames:
            prefix = name[0:len(newName)]

            if prefix == newName:
                dups += 1
        return dups
 
    def renameEmbedding(self, currentName, newName):
        if newName == currentName:
            return

        dups = self._countDuplicates(newName)
        if dups > 0:
            newName += str(dups)

        for i, name in enumerate(self.knownNames):
            if name == currentName:
                self.knownNames[i] = newName 
    
    def insertEmbedding(self, embedding, name):
        dups = self._countDuplicates(name)
        if dups > 0:
            name += str(dups)

        self.knownFaceEmbeddings.append(embedding)
        self.knownNames.append(name)

        assert len(self.knownFaceEmbeddings) == len(self.knownNames)

    def _reset(self):
        self.knownFaceEmbeddings = list()
        self.knownNames = list()



def sanityTestMemory():
    mem = FaceDetectorMemory()

    mem.insertEmbedding(0, "marius")
    mem.insertEmbedding(2, "max")
    mem.insertEmbedding(7, "Max")
    mem.insertEmbedding(7, "maX")
    mem.insertEmbedding(8, "tim")
    mem.insertEmbedding(2, "max")
    mem.insertEmbedding(3, "max")
    
    print(mem.knownNames)
    mem.renameEmbedding("maX", "marius")
    print(mem.knownNames)
    mem.renameEmbedding("marius1", "max")
    print(mem.knownNames)
    mem.renameEmbedding("marius", "marius")
    print(mem.knownNames)
    mem.renameEmbedding("marius", "tim")
    print(mem.knownNames)
    print(mem.knownFaceEmbeddings)

    mem.save("data.pkl")
    
    print("loading...")
    newMem = FaceDetectorMemory(picklePath="data.pkl")
    print(newMem.knownNames)
    print(newMem.knownFaceEmbeddings)


    

def main():
    video_capture = cv2.VideoCapture(0)
    memory = FaceDetectorMemory(picklePath="data.pkl")
    # memory = FaceDetectorMemory()
    
    MIN_NUM_OF_OCCURENCES = 60
    candidateBuffer = list()
    candidateOccurenceCnt = list()


    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(memory.knownFaceEmbeddings, encoding)
            
            if True in matches: 
                idx = matches.index(True)
                name = memory.knownNames[idx]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                # check if it matches any of the candidates 
                cndMatches = face_recognition.compare_faces(candidateBuffer, encoding)
                if True in cndMatches:
                    idx = cndMatches.index(True)
                    candidateOccurenceCnt[idx] += 1
                    if candidateOccurenceCnt[idx] > MIN_NUM_OF_OCCURENCES:
                        # save this one in memory 
                        memory.insertEmbedding(encoding, "Unknown")
                else: 
                    candidateBuffer.append(encoding)
                    candidateOccurenceCnt.append(0)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    memory.save("data.pkl")


if __name__ == "__main__":
    main()
    # sanityTestMemory()
    
    # mem = FaceDetectorMemory(picklePath="data.pkl")
    # mem.renameEmbedding("Unknown", "marius")
    # mem.renameEmbedding("Unknown1", "jonsnow")
    # mem.renameEmbedding("Unknown2", "youknownothing")
    # mem.save(picklePath="data.pkl")

